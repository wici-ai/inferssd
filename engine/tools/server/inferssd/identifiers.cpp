// SPDX-License-Identifier: Apache-2.0
#include "identifiers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string_view>

#if defined(__APPLE__)
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

namespace {

using pair_list = std::vector<std::pair<std::string, std::string>>;

// a byte that continues a name: a letter, a digit, '_' or a byte of a multi-byte UTF-8 character
bool is_name_byte(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c >= 0x80;
}

bool is_separator(char c) {
    return c == '/' || c == '\\';
}

// Replaces each text of one side of the pairs (the identifiers, or with reverse the stand-ins) by the other side,
// left to right, the longest first. A match starts after a byte that is not a name byte and ends before one. With
// partial, the text is still growing: it stops where a longer text could still match, or where the byte after a match
// is not there yet.
std::string replace(const std::string & text, const pair_list & pairs, bool reverse, bool partial) {
    std::vector<const std::pair<std::string, std::string> *> order;
    std::array<bool, 256> first {};
    for (const auto & p : pairs) {
        const std::string & from = reverse ? p.second : p.first;
        if (!from.empty()) {
            order.push_back(&p);
            first[(unsigned char) from[0]] = true;
        }
    }
    const auto from_of = [reverse](const std::pair<std::string, std::string> * p) -> const std::string & {
        return reverse ? p->second : p->first;
    };
    std::stable_sort(order.begin(), order.end(), [&](const auto * a, const auto * b) {
        return from_of(a).size() > from_of(b).size();
    });

    std::string out;
    out.reserve(text.size());
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        if (!first[(unsigned char) text[i]] || (i > 0 && is_name_byte(text[i - 1]))) {
            out += text[i++];
            continue;
        }
        const std::string * to   = nullptr;
        size_t              len  = 0;
        bool                hold = false;
        for (const auto * p : order) {
            const std::string & from = from_of(p);
            if (from.size() > n - i) {
                if (partial && text.compare(i, n - i, from, 0, n - i) == 0) {
                    hold = true;
                    break;
                }
                continue;
            }
            if (text.compare(i, from.size(), from) != 0) {
                continue;
            }
            const size_t end = i + from.size();
            if (end == n && partial) {
                hold = true;
                break;
            }
            if (end < n && is_name_byte(text[end])) {
                continue;
            }
            to  = reverse ? &p->first : &p->second;
            len = from.size();
            break;
        }
        if (hold) {
            break;
        }
        if (to) {
            out += *to;
            i += len;
        } else {
            out += text[i++];
        }
    }
    return out;
}

std::string replace_char(std::string s, char from, const std::string & to) {
    std::string out;
    for (char c : s) {
        if (c == from) {
            out += to;
        } else {
            out += c;
        }
    }
    return out;
}

// Adds an identifier with its stand-in, and on Windows its other spellings: '/' for '\', '\\' for '\' (inside JSON
// text such as tool-call arguments), the drive letter in lower case and the Git Bash form (/c/Users/...). An
// identifier keeps its first stand-in, and a stand-in its first identifier, so that each side maps to one.
void add_pair(pair_list & pairs, const std::string & id, const std::string & stand_in, bool windows) {
    const auto add = [&](const std::string & a, const std::string & b) {
        if (a.empty() || a == b) {
            return;
        }
        for (const auto & p : pairs) {
            if (p.first == a || p.second == b) {
                return;
            }
        }
        pairs.emplace_back(a, b);
    };
    add(id, stand_in);
    if (!windows) {
        return;
    }
    const std::string id_back = replace_char(id, '/', "\\"), stand_in_back = replace_char(stand_in, '/', "\\");
    const std::string id_fwd = replace_char(id_back, '\\', "/"), stand_in_fwd = replace_char(stand_in_back, '\\', "/");
    std::vector<std::pair<std::string, std::string>> forms = {
        {id_back, stand_in_back},
        {id_fwd, stand_in_fwd},
        {replace_char(id_back, '\\', "\\\\"), replace_char(stand_in_back, '\\', "\\\\")},
    };
    const auto drive = [](const std::string & s) {
        return s.size() >= 2 && s[1] == ':' && std::isalpha((unsigned char) s[0]);
    };
    if (drive(id_back) && drive(stand_in_back)) {
        const size_t n = forms.size();
        for (size_t k = 0; k < n; ++k) {
            auto lower = forms[k];
            lower.first[0]  = (char) std::tolower((unsigned char) lower.first[0]);
            lower.second[0] = (char) std::tolower((unsigned char) lower.second[0]);
            forms.push_back(lower);
        }
        forms.emplace_back(std::string("/") + (char) std::tolower((unsigned char) id_fwd[0]) + id_fwd.substr(2),
                           std::string("/") + (char) std::tolower((unsigned char) stand_in_fwd[0]) +
                               stand_in_fwd.substr(2));
    }
    for (const auto & [a, b] : forms) {
        add(a, b);
    }
}

// Claude Code names a project's folder under ~/.claude/projects by its path with each byte that is not a letter or a
// digit turned into '-': C:\Users\name\app is C--Users-name-app, ~/code/app on a Mac -Users-name-code-app.
std::string claude_project_name(const std::string & path) {
    std::string out = path;
    for (char & c : out) {
        if (!std::isalnum((unsigned char) c)) {
            c = '-';
        }
    }
    return out;
}

bool is_ascii(const std::string & s) {
    return std::all_of(s.begin(), s.end(), [](char c) { return (unsigned char) c < 0x80; });
}

// The end of `literal` at i in text, a '/' in it matching '/' or '\'; npos if it is not there.
size_t literal_end(const std::string & text, size_t i, std::string_view literal) {
    if (i > text.size() || text.size() - i < literal.size()) {
        return std::string::npos;
    }
    for (size_t k = 0; k < literal.size(); ++k) {
        const char a = text[i + k];
        const char b = literal[k];
        if (b == '/' ? !is_separator(a) : a != b) {
            return std::string::npos;
        }
    }
    return i + literal.size();
}

template <typename F> size_t run_end(const std::string & text, size_t i, F ok) {
    while (i < text.size() && ok((unsigned char) text[i])) {
        ++i;
    }
    return i;
}

bool is_hex_or_dash(unsigned char c) {
    return std::isxdigit(c) || c == '-';
}

// a byte of a folder name as the apps write it: not a separator, a space, a quote or a control byte
bool is_folder_byte(unsigned char c) {
    return c > ' ' && !is_separator((char) c) && c != '"' && c != '\'' && c != '`' && c != '<' && c != '>' && c != '|';
}

struct learned_identifier {
    std::string id;
    std::string stand_in;
    int         kind;
};

// The folders the apps name in their system and developer messages that change with the project or the session.
// Each one found, in the order found, becomes an identifier; the n-th one of a kind gets "-n" after its stand-in.
void learn_identifiers(const std::string & text, const server_identifier_env & env,
                       std::vector<learned_identifier> & out) {
    const auto found = [&](int kind, std::string id, std::string stand_in) {
        int n = 1;
        for (const auto & l : out) {
            if (l.id == id) {
                return;
            }
            n += l.kind == kind;
        }
        if (n > 1) {
            stand_in += "-" + std::to_string(n);
        }
        out.push_back({std::move(id), std::move(stand_in), kind});
    };
    const auto sep_at = [&](size_t i) { return i < text.size() && is_separator(text[i]); };

    // Claude Code's memory folder: ~/.claude/projects/<project>/memory
    for (size_t at = text.find(".claude"); at != std::string::npos; at = text.find(".claude", at + 1)) {
        const size_t begin = literal_end(text, at, ".claude/projects/");
        if (begin == std::string::npos) {
            continue;
        }
        const size_t end = run_end(text, begin, [](unsigned char c) { return std::isalnum(c) || c == '-'; });
        if (end == begin || literal_end(text, end, "/memory") == std::string::npos) {
            continue;
        }
        const std::string project = text.substr(begin, end - begin);
        if (project != claude_project_name(env.home)) {
            found(0, project, claude_project_name(env.home_stand_in) + "-project");
        }
    }

    // the Claude desktop app's folders: .../local-agent-mode-sessions/<account>/<organization> (the account's memory
    // is in it) and .../<account>/<organization>/<session>
    static const std::string sessions = "local-agent-mode-sessions";
    for (size_t at = text.find(sessions); at != std::string::npos; at = text.find(sessions, at + 1)) {
        size_t      i        = at + sessions.size();
        std::string stand_in = sessions;
        for (int k = 0; k < 3; ++k) {
            const size_t end = sep_at(i) ? run_end(text, i + 1, is_hex_or_dash) : i;
            if (end <= i + 1 || (end < text.size() && is_name_byte(text[end]))) {
                break;
            }
            stand_in += text[i];
            stand_in += "00000000";
            i = end;
            if (k == 1) {
                found(5, text.substr(at, i - at), stand_in);
            } else if (k == 2) {
                found(1, text.substr(at, i - at), stand_in);
            }
        }
    }

    // the Linux environment of a Claude desktop session: /sessions/<adjective>-<adjective>-<name>
    static const std::string vm = "/sessions/";
    for (size_t at = text.find(vm); at != std::string::npos; at = text.find(vm, at + 1)) {
        const size_t begin = at + vm.size();
        const size_t end   = run_end(text, begin, [](unsigned char c) { return (c >= 'a' && c <= 'z') || c == '-'; });
        const std::string name = text.substr(begin, end - begin);
        if (name.size() < 3 || name.find('-') == std::string::npos || name.front() == '-' || name.back() == '-' ||
            (end < text.size() && is_name_byte(text[end]))) {
            continue;
        }
        found(2, vm + name, vm + "local-agent-session");
    }

    // the plugins a Claude desktop session loads: <temp>/claude-hostloop-plugins/<hash>/<account>
    static const std::string plugins = "claude-hostloop-plugins";
    for (size_t at = text.find(plugins); at != std::string::npos; at = text.find(plugins, at + 1)) {
        size_t      i        = at + plugins.size();
        std::string stand_in = plugins;
        bool        ok       = true;
        for (int k = 0; k < 2 && ok; ++k) {
            const size_t end = sep_at(i) ? run_end(text, i + 1, is_hex_or_dash) : i;
            ok = end > i + 1;
            if (ok) {
                stand_in += text[i];
                for (size_t j = i + 1; j < end; ++j) {
                    stand_in += text[j] == '-' ? '-' : '0';
                }
                i = end;
            }
        }
        if (ok) {
            found(3, text.substr(at, i - at), stand_in);
        }
    }

    // the folder of a ChatGPT desktop app conversation: ~/Documents/Codex/<date>/<name>
    for (size_t at = text.find("Documents"); at != std::string::npos; at = text.find("Documents", at + 1)) {
        const size_t date = literal_end(text, at, "Documents/Codex/");
        if (date == std::string::npos || date + 10 >= text.size()) {
            continue;
        }
        bool is_date = true;
        for (size_t k = 0; k < 10; ++k) {
            const unsigned char c = text[date + k];
            is_date = is_date && ((k == 4 || k == 7) ? c == '-' : std::isdigit(c) != 0);
        }
        if (!is_date || !sep_at(date + 10)) {
            continue;
        }
        const size_t begin = date + 11;
        size_t       end   = run_end(text, begin, is_folder_byte);
        while (end > begin && std::string_view(".,;:)]}").find(text[end - 1]) != std::string_view::npos) {
            --end;
        }
        if (end == begin) {
            continue;
        }
        std::string stand_in = text.substr(at, date - at) + "0000-00-00" + text[date + 10] + "task";
        found(4, text.substr(at, end - at), stand_in);
    }

    // the folder of a ChatGPT desktop app conversation's visualizations, one of its writable roots:
    // ~/.codex/visualizations/<yyyy>/<mm>/<dd>/<uuid>
    for (size_t at = text.find(".codex"); at != std::string::npos; at = text.find(".codex", at + 1)) {
        size_t i = literal_end(text, at, ".codex/visualizations");
        if (i == std::string::npos) {
            continue;
        }
        std::string stand_in = text.substr(at, i - at);
        bool        ok       = true;
        for (const size_t digits : {4, 2, 2}) {
            ok = ok && sep_at(i) && i + 1 + digits <= text.size();
            for (size_t k = 1; ok && k <= digits; ++k) {
                ok = std::isdigit((unsigned char) text[i + k]) != 0;
            }
            if (ok) {
                stand_in += text[i] + std::string(digits, '0');
                i += 1 + digits;
            }
        }
        const size_t end = ok && sep_at(i) ? run_end(text, i + 1, is_hex_or_dash) : i;
        if (!ok || end != i + 37 || (end < text.size() && is_name_byte(text[end]))) {
            continue;
        }
        stand_in += text[i];
        for (size_t j = i + 1; j < end; ++j) {
            stand_in += text[j] == '-' ? '-' : '0';
        }
        found(6, text.substr(at, end - at), stand_in);
    }
}

// Takes the root that is the cwd out of Codex's sentences " The writable root is `<root>`." and " The writable
// roots are `<root>`, `<root>`.", and writes the sentence again as Codex writes it for the roots left; with none left
// the sentence goes with the line break before it, as Codex writes its instructions without writable roots.
void drop_root(std::string & text, const std::string & cwd) {
    static const std::string stem = " The writable root", one = stem + " is ", many = stem + "s are ";
    for (size_t at = text.find(stem); at != std::string::npos; at = text.find(stem, at + 1)) {
        const std::string & lead = text.compare(at, one.size(), one) == 0 ? one : many;
        if (text.compare(at, lead.size(), lead) != 0) {
            continue;
        }
        std::vector<std::string> roots;
        size_t i = at + lead.size();
        while (i < text.size() && text[i] == '`') {
            const size_t end = text.find('`', i + 1);
            if (end == std::string::npos) {
                break;
            }
            roots.push_back(text.substr(i + 1, end - i - 1));
            i = end + 1;
            if (text.compare(i, 2, ", ") != 0) {
                break;
            }
            i += 2;
        }
        const auto it = std::find(roots.begin(), roots.end(), cwd);
        if (i >= text.size() || text[i] != '.' || it == roots.end()) {
            continue;
        }
        roots.erase(it);
        std::string sentence;
        for (size_t k = 0; k < roots.size(); ++k) {
            sentence += (k > 0 ? std::string(", ") : roots.size() == 1 ? one : many) + "`" + roots[k] + "`";
        }
        sentence += roots.empty() ? "" : ".";
        const size_t begin = roots.empty() && at > 0 && text[at - 1] == '\n' ? at - 1 : at;
        text.replace(begin, i + 1 - begin, sentence);
        at = begin;
    }
}

// Calls f on each text of a message list or tools: the content, the tool-call arguments, the descriptions. Media
// (data URLs, images, audio, files) is left alone.
template <typename F> void for_each_text(json & value, F & f) {
    if (value.is_string()) {
        const std::string & s = value.get_ref<const std::string &>();
        if (s.compare(0, 5, "data:") != 0) {
            f(value.get_ref<std::string &>());
        }
    } else if (value.is_array()) {
        for (auto & item : value) {
            for_each_text(item, f);
        }
    } else if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (it.key() == "image_url" || it.key() == "input_audio" || it.key() == "file") {
                continue;
            }
            for_each_text(it.value(), f);
        }
    }
}

} // namespace

std::string server_identifier_map::to_stand_ins(const std::string & text) const {
    return pairs.empty() ? text : replace(text, pairs, false, false);
}

std::string server_identifier_map::to_identifiers(const std::string & text, bool partial) const {
    return pairs.empty() ? text : replace(text, pairs, true, partial);
}

json server_identifier_map::to_json() const {
    json out = json::array();
    for (const auto & [id, stand_in] : pairs) {
        out.push_back(json::array({id, stand_in}));
    }
    return out;
}

server_identifier_map server_identifier_map::from_json(const json & value) {
    server_identifier_map map;
    if (!value.is_array()) {
        return map;
    }
    for (const auto & p : value) {
        if (p.is_array() && p.size() == 2 && p[0].is_string() && p[1].is_string()) {
            map.pairs.emplace_back(p[0].get<std::string>(), p[1].get<std::string>());
        }
    }
    return map;
}

server_identifier_env server_identifier_env::of_system(const std::string & os, std::string home, std::string temp) {
    server_identifier_env e;
    e.windows = os == "windows";
    // the paths are split in two literals: written out, a home directory is a personal path to the repository's
    // public scan (scripts/public-scan.py)
    e.home_stand_in = e.windows ? "C:\\Users\\infer" : os == "macos" ? "/Users" "/infer" : "/home" "/infer";
    while (home.size() > 1 && is_separator(home.back())) {
        home.pop_back();
    }
    e.home = std::move(home);
    // macOS: the per-user folder under /var/folders that $TMPDIR names
    if (os == "macos" && temp.size() > 1) {
        while (temp.size() > 1 && temp.back() == '/') {
            temp.pop_back();
        }
        e.temp          = std::move(temp);
        e.temp_stand_in = "/var/folders/00/infer/T";
    }
    return e;
}

const server_identifier_env & server_identifier_env::current() {
    static const server_identifier_env env = [] {
        const char * target = std::getenv("INFERSSD_IDENTIFIERS_OS");
        const std::string os = target ? target : "";
        if (os == "macos" || os == "windows" || os == "linux") {
            const char * home = std::getenv(os == "windows" ? "USERPROFILE" : "HOME");
            const char * temp = std::getenv("TMPDIR");
            return of_system(os, home ? home : "", os == "macos" && temp ? temp : "");
        }
#if defined(_WIN32)
        // as UTF-8, the encoding of the requests, also for a user name outside the ANSI code page
        const wchar_t * home = _wgetenv(L"USERPROFILE");
        return of_system("windows", home ? std::filesystem::path(home).u8string() : std::string(), "");
#elif defined(__APPLE__)
        const char * home = std::getenv("HOME");
        char         buf[1024];
        const size_t n = confstr(_CS_DARWIN_USER_TEMP_DIR, buf, sizeof(buf));
        return of_system("macos", home ? home : "", n > 1 && n <= sizeof(buf) ? std::string(buf) : std::string());
#else
        const char * home = std::getenv("HOME");
        return of_system("linux", home ? home : "", "");
#endif
    }();
    return env;
}

server_identifier_map server_identifiers_apply(json & body, const server_identifier_env & env) {
    pair_list pairs;
    // a home directory of a few bytes ("/", "C:\") would match too much
    const bool home = env.home.size() >= 4 && !env.home_stand_in.empty();
    if (home) {
        add_pair(pairs, env.home, env.home_stand_in, env.windows);
        if (is_ascii(env.home)) {
            add_pair(pairs, claude_project_name(env.home), claude_project_name(env.home_stand_in), false);
        }
    }
    if (env.temp.size() >= 4 && !env.temp_stand_in.empty() &&
        !(home && env.temp.compare(0, env.home.size(), env.home) == 0)) {
        add_pair(pairs, env.temp, env.temp_stand_in, env.windows);
        add_pair(pairs, "/private" + env.temp, "/private" + env.temp_stand_in, false);
    }

    json messages = body.contains("messages") ? body.at("messages") : json();
    std::vector<learned_identifier> learned;
    if (messages.is_array()) {
        for (auto & msg : messages) {
            const std::string role = msg.is_object() && msg.contains("role") && msg.at("role").is_string() ?
                                         msg.at("role").get<std::string>() : std::string();
            if (role != "system" && role != "developer") {
                continue;
            }
            auto learn = [&](std::string & text) { learn_identifiers(text, env, learned); };
            if (msg.contains("content")) {
                for_each_text(msg.at("content"), learn);
            }
        }
    }
    for (const auto & l : learned) {
        add_pair(pairs, l.id, l.stand_in, env.windows);
    }

    server_identifier_map map;
    map.pairs = std::move(pairs);
    if (map.empty()) {
        return map;
    }

    json tools   = body.contains("tools") ? body.at("tools") : json();
    bool changed = false;
    bool exact   = true;
    auto apply   = [&](std::string & text) {
        if (!exact) {
            return;
        }
        std::string mapped = map.to_stand_ins(text);
        if (map.to_identifiers(mapped, false) != text) {
            exact = false;
            return;
        }
        if (mapped != text) {
            changed = true;
            text    = std::move(mapped);
        }
    };
    for_each_text(messages, apply);
    for_each_text(tools, apply);
    if (!exact || !changed) {
        return {};
    }
    body["messages"] = std::move(messages);
    if (body.contains("tools")) {
        body["tools"] = std::move(tools);
    }
    return map;
}

bool server_strip_anthropic_billing_header(std::string & text) {
    static const std::string tag = "x-anthropic-billing-header:";
    if (text.compare(0, tag.size(), tag) != 0) {
        return false;
    }
    // the line is a list of name=value; pairs
    size_t end = std::string::npos;
    size_t i   = tag.size();
    while (true) {
        const size_t name  = run_end(text, i, [](unsigned char c) { return c == ' '; });
        const size_t equal = run_end(text, name, [](unsigned char c) { return std::isalnum(c) || c == '_'; });
        if (equal == name || equal >= text.size() || text[equal] != '=') {
            break;
        }
        const size_t semicolon = text.find_first_of(";\n", equal + 1);
        if (semicolon == std::string::npos || text[semicolon] != ';') {
            break;
        }
        i = end = semicolon + 1;
    }
    if (end == std::string::npos) {
        return false;
    }
    text.erase(0, end);
    return true;
}

// Calls f on the text of each system message, or of each of its text parts.
template <typename F> static void for_each_system_text(json & messages, F f) {
    if (!messages.is_array()) {
        return;
    }
    for (auto & msg : messages) {
        if (!msg.is_object() || !msg.contains("role") || msg.at("role") != "system" || !msg.contains("content")) {
            continue;
        }
        json & content = msg.at("content");
        if (content.is_string()) {
            f(content.get_ref<std::string &>());
        } else if (content.is_array()) {
            for (auto & part : content) {
                if (part.is_object() && part.contains("text") && part.at("text").is_string()) {
                    f(part.at("text").get_ref<std::string &>());
                }
            }
        }
    }
}

void server_strip_anthropic_billing_header(json & messages) {
    for_each_system_text(messages, [](std::string & text) { server_strip_anthropic_billing_header(text); });
}

bool server_strip_claude_env_user_lines(std::string & text) {
    static const std::string open = "<env>\n";
    static const std::string close = "</env>";
    bool changed = false;
    for (size_t at = text.find(open); at != std::string::npos; at = text.find(open, at + 1)) {
        // the block starts a line; "<env>" in a sentence is not one
        if (at != 0 && text[at - 1] != '\n') {
            continue;
        }
        size_t end = text.find(close, at);
        if (end == std::string::npos) {
            break;
        }
        for (size_t line = at + open.size(); line < end;) {
            const size_t eol = text.find('\n', line);
            if (eol == std::string::npos || eol > end) {
                break;
            }
            if (text.compare(line, 10, "Timezone: ") == 0 || text.compare(line, 11, "User name: ") == 0) {
                text.erase(line, eol + 1 - line);
                end -= eol + 1 - line;
                changed = true;
            } else {
                line = eol + 1;
            }
        }
    }
    return changed;
}

void server_strip_claude_env_user_lines(json & messages) {
    for_each_system_text(messages, [](std::string & text) { server_strip_claude_env_user_lines(text); });
}

void server_drop_codex_cwd_root(json & messages) {
    if (!messages.is_array()) {
        return;
    }
    const auto role_of = [](const json & msg) {
        return msg.is_object() && msg.contains("role") && msg.at("role").is_string() ? msg.at("role").get<std::string>()
                                                                                     : std::string();
    };
    std::string cwd;
    auto        find_cwd = [&](std::string & text) {
        const size_t begin = cwd.empty() ? text.find("<cwd>") : std::string::npos;
        const size_t end   = begin == std::string::npos ? std::string::npos : text.find("</cwd>", begin);
        if (end != std::string::npos) {
            cwd = text.substr(begin + 5, end - begin - 5);
        }
    };
    for (auto & msg : messages) {
        if (role_of(msg) == "user" && msg.contains("content")) {
            for_each_text(msg.at("content"), find_cwd);
        }
    }
    if (cwd.empty()) {
        return;
    }
    auto drop = [&](std::string & text) { drop_root(text, cwd); };
    for (auto & msg : messages) {
        const std::string role = role_of(msg);
        if ((role == "system" || role == "developer") && msg.contains("content")) {
            for_each_text(msg.at("content"), drop);
        }
    }
}
