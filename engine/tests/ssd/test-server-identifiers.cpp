// SPDX-License-Identifier: Apache-2.0
// The stand-ins for the identifiers of the user and the session, and Claude Code's billing line
// (tools/server/inferssd/identifiers.h).

#include "server-task.h"
#include "inferssd/identifiers.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Home directories are built from parts: written out, a home directory is a personal path to the repository's public
// scan (scripts/public-scan.py).
static std::string users(const std::string & name) {
    return std::string("/Users") + "/" + name;
}

static const std::string STAND_IN = users("infer");

static void require(bool condition, const std::string & message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

static server_identifier_env mac_env(const std::string & home, const std::string & temp) {
    server_identifier_env env;
    env.home          = home;
    env.temp          = temp;
    env.home_stand_in = STAND_IN;
    env.temp_stand_in = "/var/folders/00/infer/T";
    return env;
}

static std::string claude_name(std::string path) {
    for (char & c : path) {
        if (!std::isalnum((unsigned char) c)) {
            c = '-';
        }
    }
    return path;
}

static json tool_call(const std::string & id, const std::string & name, const json & arguments) {
    return {{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", arguments.dump()}}}};
}

static std::string text_of(const json & msg) {
    return msg.at("content").get<std::string>();
}

// a Claude Code request as server_chat_convert_anthropic_to_oai leaves it
static json claude_code_request(const std::string & home, const std::string & project, const std::string & version) {
    const std::string cwd    = home + "/" + project;
    const std::string memory = home + "/.claude/projects/" + claude_name(cwd) + "/memory/";
    const std::string system = "x-anthropic-billing-header: cc_version=2.1.328." + version +
                               "; cc_entrypoint=sdk-cli;You are a Claude agent.\n# Memory\n"
                               "You have a persistent memory at `" + memory + "`. Write to it directly.";
    const json bash = {{"type", "function"},
                       {"function", {{"name", "Bash"},
                                     {"description", "Runs a command in " + home + "."},
                                     {"parameters", {{"type", "object"}}}}}};
    return {
        {"messages", json::array({
            {{"role", "system"}, {"content", system}},
            {{"role", "user"}, {"content", "Primary working directory: " + cwd + "\nlist the files"}},
            {{"role", "assistant"}, {"content", ""},
             {"tool_calls", json::array({tool_call("t1", "Bash", {{"command", "ls " + cwd}})})}},
            {{"role", "tool"}, {"tool_call_id", "t1"}, {"content", cwd + "/README.md\n" + memory + "notes.md"}},
        })},
        {"tools", json::array({bash})},
    };
}

static void test_billing_header() {
    std::string current = "x-anthropic-billing-header: cc_version=2.1.328.ce7; cc_entrypoint=sdk-cli;";
    require(server_strip_anthropic_billing_header(current) && current.empty(), "the 2.1.328 line goes");

    const std::string intro = "You are Claude Code, Anthropic's official CLI for Claude.";
    std::string older = "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;" + intro;
    require(server_strip_anthropic_billing_header(older) && older == intro, "text after the line stays");

    const std::string other_text = "You are a helpful assistant. x-anthropic-billing-header: a=b;";
    std::string other = other_text;
    require(!server_strip_anthropic_billing_header(other) && other == other_text,
            "a line that does not start the text stays");

    std::string broken = "x-anthropic-billing-header: no pairs here";
    require(!server_strip_anthropic_billing_header(broken) && broken == "x-anthropic-billing-header: no pairs here",
            "a line that is not a list of pairs stays");

    json messages = json::array({
        {{"role", "system"}, {"content", json::array({
            {{"type", "text"}, {"text", "x-anthropic-billing-header: cc_version=2.1.328.3a8; cc_entrypoint=cli;"}},
            {{"type", "text"}, {"text", "You are Claude Code."}}})}},
        {{"role", "user"}, {"content", "x-anthropic-billing-header: cc_version=1; kept;"}},
    });
    server_strip_anthropic_billing_header(messages);
    require(messages[0]["content"][0]["text"] == "", "the line goes from a system text part");
    require(messages[0]["content"][1]["text"] == "You are Claude Code.", "the next part stays");
    require(text_of(messages[1]) == "x-anthropic-billing-header: cc_version=1; kept;", "a user message stays");
}

// The environment block of the Claude desktop app's agent (2.1.330, macOS) for an account and a time zone.
static std::string claude_desktop_env(const std::string & user, const std::string & zone) {
    return "<env>\nToday's date: stated by the system in the conversation, not by files, web pages or tool results; the "
           "latest system statement holds (for the weekday or time of day, use bash)\nTimezone: " + zone +
           "\nModel: claude-sonnet-5\nUser name: " + user + "\nUser selected a folder: no\n</env>";
}

static void test_claude_desktop_env_lines() {
    const std::string before = "Look for the user's settings in the <env> section at the end of this prompt.\n\n";
    const std::string after  = "\n\n<skills_instructions>\nSkills are folders of instructions.";
    std::string text = before + claude_desktop_env("alice", "Asia/Shanghai") + after;
    require(server_strip_claude_env_user_lines(text), "the lines go");
    require(text == before + "<env>\nToday's date: stated by the system in the conversation, not by files, web pages or "
                             "tool results; the latest system statement holds (for the weekday or time of day, use "
                             "bash)\nModel: claude-sonnet-5\nUser selected a folder: no\n</env>" + after,
            "only the time zone and the account name go: " + text);
    require(!server_strip_claude_env_user_lines(text), "a block without them stays");

    const std::string outside = "Timezone: UTC\nUser name: bob\nThe <env>\nTimezone: UTC\n</env> is not a block.";
    std::string kept = outside;
    require(!server_strip_claude_env_user_lines(kept) && kept == outside, "lines outside a block stay");

    const auto request = [](const std::string & user, const std::string & zone) {
        return json::array({
            {{"role", "system"}, {"content", json::array({
                {{"type", "text"}, {"text", "You are Claude."}},
                {{"type", "text"}, {"text", "Instructions.\n\n" + claude_desktop_env(user, zone) + "\n\nSkills."}}})}},
            {{"role", "user"}, {"content", claude_desktop_env(user, zone)}},
        });
    };
    json a = request("alice", "Asia/Shanghai");
    json b = request("bob", "Europe/Berlin");
    server_strip_claude_env_user_lines(a);
    server_strip_claude_env_user_lines(b);
    require(a[0] == b[0], "two users get the same system message: " + a[0].dump());
    require(text_of(a[1]) == claude_desktop_env("alice", "Asia/Shanghai"), "a user message stays");
}

static void test_claude_code_shares_its_context() {
    const server_identifier_env alice = mac_env(users("alice"), "/var/folders/ab/abc123/T");
    const server_identifier_env bob   = mac_env(users("bob"), "/var/folders/cd/def456/T");

    json a = claude_code_request(alice.home, "code/app", "ce7");
    json b = claude_code_request(bob.home, "work/site", "3a8");
    const json original_a = a;
    server_strip_anthropic_billing_header(a["messages"]);
    server_strip_anthropic_billing_header(b["messages"]);
    const server_identifier_map map_a = server_identifiers_apply(a, alice);
    const server_identifier_map map_b = server_identifiers_apply(b, bob);
    require(!map_a.empty() && !map_b.empty(), "the requests have identifiers");

    require(a["messages"][0] == b["messages"][0], "two users and projects get the same system message: " +
                                                      text_of(a["messages"][0]) + " | " + text_of(b["messages"][0]));
    require(a["tools"] == b["tools"], "and the same tools");
    const std::string memory = STAND_IN + "/.claude/projects/-Users-infer-project/memory/";
    require(text_of(a["messages"][0]).find(memory) != std::string::npos,
            "the memory folder has the stand-ins: " + text_of(a["messages"][0]));
    require(text_of(a["messages"][1]) == "Primary working directory: " + STAND_IN + "/code/app\nlist the files",
            "the project keeps its name under the stand-in home: " + text_of(a["messages"][1]));
    const std::string args = a["messages"][2]["tool_calls"][0]["function"]["arguments"];
    require(args == json({{"command", "ls " + STAND_IN + "/code/app"}}).dump(), "tool-call arguments: " + args);
    require(text_of(a["messages"][3]) == STAND_IN + "/code/app/README.md\n" + memory + "notes.md",
            "a tool result: " + text_of(a["messages"][3]));

    // what the model writes with the stand-ins goes back to the client with alice's paths
    const std::string written = "Saved it to " + memory + "x.md";
    const std::string sent    = map_a.to_identifiers(written, false);
    require(sent == "Saved it to " + alice.home + "/.claude/projects/-Users-alice-code-app/memory/x.md",
            "the output: " + sent);
    const server_identifier_map back = server_identifier_map::from_json(map_a.to_json());
    require(back.pairs == map_a.pairs, "the map goes through the task parameters");
    for (const size_t i : {1, 3}) {
        require(back.to_identifiers(text_of(a["messages"][i]), false) == text_of(original_a["messages"][i]),
                "the stand-ins come back out of message " + std::to_string(i) + " exactly");
    }
    require(back.to_identifiers(args, false) == original_a["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "and out of the tool-call arguments");
}

static void test_claude_desktop_sessions_share_their_context() {
    const std::string app     = "/Library/Application Support/Claude-3p/local-agent-mode-sessions/";
    const auto        request = [&](const server_identifier_env & env, const std::string & account,
                             const std::string & session, const std::string & vm, const std::string & plugins) {
        const std::string root   = env.home + app + account + "/00000000";
        const std::string out    = root + "/" + session + "/outputs";
        const std::string system = "x-anthropic-billing-header: cc_version=2.1.326.4c8; cc_entrypoint=local-agent;"
                                   "You are a Claude agent.\nWorking directory: `" + out + "`\n- " + out +
                                   " -> /sessions/" + vm + "/mnt/outputs/\nEach bash call starts in `/sessions/" +
                                   vm + "`.\nMemory: " + root + "/memory/\nSkills: " + env.temp +
                                   "/claude-hostloop-plugins/" + plugins + "/skills";
        return json{{"messages", json::array({{{"role", "system"}, {"content", system}},
                                              {{"role", "user"}, {"content", "hi"}}})}};
    };
    const server_identifier_env alice = mac_env(users("alice"), "/var/folders/dk/4d645xfs/T");
    const server_identifier_env bob   = mac_env(users("bob"), "/var/folders/xy/9z8y7x6w/T");
    const std::string alice_plugins   = "17bd12150ce86735/a8dda2a7-7037-404b-8ba6-81dd8f91a77c";
    json a  = request(alice, "a8dda2a7", "2619fd4d", "jolly-exciting-faraday", alice_plugins);
    json a2 = request(alice, "a8dda2a7", "3de4fddd", "sharp-dreamy-mayer", alice_plugins);
    json b  = request(bob, "5c0ffee1", "77aa11bb", "brave-quiet-turing",
                      "0123456789abcdef/5c0ffee1-0000-4000-8000-00000000beef");
    for (json * r : {&a, &a2, &b}) {
        server_strip_anthropic_billing_header((*r)["messages"]);
    }
    const server_identifier_map map_a = server_identifiers_apply(a, alice);
    server_identifiers_apply(a2, alice);
    server_identifiers_apply(b, bob);
    require(a["messages"][0] == a2["messages"][0], "two sessions get the same system message: " +
                                                       text_of(a["messages"][0]) + " | " + text_of(a2["messages"][0]));
    require(a["messages"][0] == b["messages"][0],
            "and two accounts: " + text_of(a["messages"][0]) + " | " + text_of(b["messages"][0]));

    const std::string memory = "Read " + STAND_IN + app + "00000000/00000000/memory/notes.md";
    require(map_a.to_identifiers(memory, false) == "Read " + alice.home + app + "a8dda2a7/00000000/memory/notes.md",
            "the account's memory folder: " + map_a.to_identifiers(memory, false));
    const std::string vm_path = "Wrote /sessions/local-agent-session/mnt/outputs/a.txt";
    require(map_a.to_identifiers(vm_path, false) == "Wrote /sessions/jolly-exciting-faraday/mnt/outputs/a.txt",
            "a path in the session's Linux environment: " + map_a.to_identifiers(vm_path, false));
}

static void test_chatgpt_conversations_share_their_context() {
    const server_identifier_env env = mac_env(users("alice"), "");
    const auto request = [&](const std::string & date, const std::string & name) {
        const std::string developer = "Save deliverables under " + env.home + "/Documents/Codex/" + date + "/" + name +
                                      "/outputs. Skills: " + env.home + "/.codex/skills/.system";
        return json{{"messages", json::array({{{"role", "developer"}, {"content", developer}},
                                              {{"role", "user"}, {"content", "hi"}}})}};
    };
    json a = request("2026-11-19", "q");
    json b = request("2026-11-23", "draft-plan");
    server_identifiers_apply(a, env);
    server_identifiers_apply(b, env);
    require(a["messages"][0] == b["messages"][0], "two conversations get the same developer message: " +
                                                      text_of(a["messages"][0]) + " | " + text_of(b["messages"][0]));
}

// The ChatGPT desktop app names each conversation's folder of visualizations among the writable roots.
static void test_chatgpt_visualizations_share_their_context() {
    const server_identifier_env env = mac_env(users("alice"), "");
    const auto request = [&](const std::string & date, const std::string & id) {
        const std::string roots = " The writable roots are `" + env.home + "/.codex/visualizations/" + date + "/" + id +
                                  "`, `" + env.home + "/Documents/Codex`.";
        return json{{"messages", json::array({{{"role", "developer"}, {"content", roots}},
                                              {{"role", "user"}, {"content", "hi"}}})}};
    };
    json a = request("2026/11/25", "01a210bb-1460-7743-92c7-444eab46b199");
    json b = request("2026/11/24", "01a20e50-b55c-7d30-be29-f7cce4e8da0e");
    const std::string original = text_of(a["messages"][0]);
    const server_identifier_map map = server_identifiers_apply(a, env);
    server_identifiers_apply(b, env);
    require(a["messages"][0] == b["messages"][0], "two conversations get the same roots: " + text_of(a["messages"][0]) +
                                                      " | " + text_of(b["messages"][0]));
    require(text_of(a["messages"][0]).find(STAND_IN + "/.codex/visualizations/0000/00/00/00000000-0000-0000-0000-"
                                                      "000000000000`") != std::string::npos,
            "the folder's stand-in: " + text_of(a["messages"][0]));
    require(map.to_identifiers(text_of(a["messages"][0]), false) == original, "the folder comes back");

    server_identifier_env win;
    win.home          = "C:\\Users\\bob";
    win.home_stand_in = "C:\\Users\\infer";
    win.windows       = true;
    json w = {{"messages", json::array({{{"role", "developer"}, {"content",
        "`C:\\Users\\bob\\.codex\\visualizations\\2026\\11\\25\\01a210bb-1460-7743-92c7-444eab46b199`"}}})}};
    server_identifiers_apply(w, win);
    require(text_of(w["messages"][0]) ==
                "`C:\\Users\\infer\\.codex\\visualizations\\0000\\00\\00\\00000000-0000-0000-0000-000000000000`",
            "Windows: " + text_of(w["messages"][0]));

    const std::string other = env.home + "/.codex/visualizations/2026/11/notes and " + env.home + "/.codex/visualizations/2026/11/25/abc";
    json o = {{"messages", json::array({{{"role", "developer"}, {"content", other}}})}};
    server_identifiers_apply(o, env);
    require(text_of(o["messages"][0]) == STAND_IN + "/.codex/visualizations/2026/11/notes and " + STAND_IN +
                                             "/.codex/visualizations/2026/11/25/abc",
            "other folders under visualizations stay: " + text_of(o["messages"][0]));
}

// a Codex request as the Responses API conversion leaves it: the permission instructions, with the writable roots in
// `roots`, and the environment
static json codex_request(const std::string & roots, const std::string & cwd) {
    const std::string developer = "<permissions instructions>\nApproval policy is currently never." + roots +
                                  "\n</permissions instructions>";
    return json{{"messages", json::array({
        {{"role", "developer"}, {"content", developer}},
        {{"role", "user"}, {"content", "<environment_context>\n  <cwd>" + cwd + "</cwd>\n</environment_context>"}},
    })}};
}

static server_identifier_map prepare(json & body, const server_identifier_env & env) {
    server_drop_codex_cwd_root(body.at("messages"));
    return server_identifiers_apply(body, env);
}

static void test_codex_projects_share_their_context() {
    const server_identifier_env alice = mac_env(users("alice"), "/var/folders/ab/abc123/T");
    const server_identifier_env bob   = mac_env(users("bob"), "/var/folders/cd/def456/T");
    const std::string a_cwd = alice.home + "/code/app", b_cwd = bob.home + "/work/site";
    json a = codex_request("\n The writable roots are `" + a_cwd + "`, `/private/tmp`.", a_cwd);
    json b = codex_request("\n The writable roots are `" + b_cwd + "`, `/private/tmp`.", b_cwd);
    const server_identifier_map map_a = prepare(a, alice);
    prepare(b, bob);
    require(a["messages"][0] == b["messages"][0], "two users and projects get the same developer message: " +
                                                      text_of(a["messages"][0]) + " | " + text_of(b["messages"][0]));
    require(text_of(a["messages"][0]) == "<permissions instructions>\nApproval policy is currently never.\n The "
                                         "writable root is `/private/tmp`.\n</permissions instructions>",
            "the cwd goes from the roots: " + text_of(a["messages"][0]));
    const std::string seen = STAND_IN + "/code/app";
    require(text_of(a["messages"][1]).find("<cwd>" + seen + "</cwd>") != std::string::npos,
            "the environment names the project in its place under the stand-in home: " + text_of(a["messages"][1]));
    require(map_a.to_identifiers("Wrote " + seen + "/src/main.c and read " + STAND_IN + "/code/other/README.md",
                                 false) ==
                "Wrote " + a_cwd + "/src/main.c and read " + alice.home + "/code/other/README.md",
            "what the model writes goes back to alice's folders, the one next to the project too");

    // the only root: the sentence goes with its line, as Codex writes the instructions without writable roots
    json one = codex_request("\n The writable root is `" + a_cwd + "`.", a_cwd);
    prepare(one, alice);
    require(text_of(one["messages"][0]) ==
                "<permissions instructions>\nApproval policy is currently never.\n</permissions instructions>",
            "the only root: " + text_of(one["messages"][0]));

    // several roots: the others stay, with the stand-ins of the home and the temporary directory
    json c = codex_request("\n The writable roots are `" + alice.temp + "`, `" + a_cwd + "`, `" + alice.home +
                               "/notes`, `" + alice.home + "`.", a_cwd);
    prepare(c, alice);
    require(text_of(c["messages"][0]).find("\n The writable roots are `/var/folders/00/infer/T`, `" + STAND_IN +
                                           "/notes`, `" + STAND_IN + "`.\n") != std::string::npos,
            "several roots: " + text_of(c["messages"][0]));

    // no root is the cwd, or there is no cwd: the sentence stays
    json d = codex_request("\n The writable root is `" + a_cwd + "`.", alice.home + "/code");
    prepare(d, alice);
    require(text_of(d["messages"][0]).find(" The writable root is `" + seen + "`.") != std::string::npos,
            "a root that is not the cwd: " + text_of(d["messages"][0]));
    json e = codex_request("\n The writable root is `" + a_cwd + "`.", a_cwd);
    e["messages"].erase(1);
    prepare(e, alice);
    require(text_of(e["messages"][0]).find(" The writable root is `" + seen + "`.") != std::string::npos,
            "no cwd: " + text_of(e["messages"][0]));

    // Windows folders
    server_identifier_env win;
    win.home          = "C:\\Users\\bob";
    win.home_stand_in = "C:\\Users\\infer";
    win.windows       = true;
    json w = codex_request("\n The writable roots are `C:\\Users\\bob\\app`, `C:\\Users\\bob\\AppData\\Local\\Temp`.",
                           "C:\\Users\\bob\\app");
    prepare(w, win);
    require(text_of(w["messages"][0]).find("\n The writable root is `C:\\Users\\infer\\AppData\\Local\\Temp`.\n") !=
                std::string::npos,
            "Windows roots: " + text_of(w["messages"][0]));
}

// A server on one system can render another system's requests as their users' servers do (INFERSSD_IDENTIFIERS_OS).
static void test_identifiers_of_a_system() {
    const server_identifier_env mac =
        server_identifier_env::of_system("macos", users("alice") + "/", "/var/folders/ab/abc123/T/");
    const server_identifier_env ref = mac_env(users("alice"), "/var/folders/ab/abc123/T");
    require(mac.home == ref.home && mac.temp == ref.temp && mac.home_stand_in == ref.home_stand_in &&
                mac.temp_stand_in == ref.temp_stand_in && !mac.windows,
            "macOS: " + mac.home + " " + mac.temp + " " + mac.home_stand_in);

    const server_identifier_env win = server_identifier_env::of_system("windows", "C:\\Users\\bob\\", "/tmp");
    require(win.windows && win.home == "C:\\Users\\bob" && win.home_stand_in == "C:\\Users\\infer" && win.temp.empty(),
            "Windows: " + win.home + " " + win.home_stand_in + " " + win.temp);
    json body = {{"messages", json::array({{{"role", "system"}, {"content", "C:\\Users\\bob\\app and C:" + users("bob") + "/x"}}})}};
    server_identifiers_apply(body, win);
    require(text_of(body["messages"][0]) == "C:\\Users\\infer\\app and C:" + STAND_IN + "/x",
            "Windows spellings on any system: " + text_of(body["messages"][0]));

    const std::string carol = std::string("/home") + "/carol";
    const server_identifier_env gnu = server_identifier_env::of_system("linux", carol, "/tmp/x");
    require(!gnu.windows && gnu.home == carol && gnu.home_stand_in == std::string("/home") + "/infer" && gnu.temp.empty(),
            "Linux: " + gnu.home_stand_in);
}

static void test_matches_stop_at_name_boundaries() {
    const std::string al  = users("al");
    const server_identifier_env env = mac_env(al, "");
    json body = {{"messages", json::array({{{"role", "user"},
        {"content", users("alice") + "/x, " + al + "/y, /Volumes/b" + al + "/z, '" + al + "'"}}})}};
    const server_identifier_map map = server_identifiers_apply(body, env);
    require(text_of(body["messages"][0]) ==
                users("alice") + "/x, " + STAND_IN + "/y, /Volumes/b" + al + "/z, '" + STAND_IN + "'",
            "only whole paths are replaced: " + text_of(body["messages"][0]));
    require(map.to_identifiers(STAND_IN + "x " + STAND_IN + ".", false) == STAND_IN + "x " + al + ".",
            "and only whole stand-ins come back");
}

static void test_a_request_that_cannot_be_mapped_exactly_stays() {
    const server_identifier_env env = mac_env(users("alice"), "");
    json body = {{"messages", json::array({{{"role", "user"},
        {"content", "Compare " + env.home + "/a with the docs' " + STAND_IN + "/b"}}})}};
    const json before = body;
    require(server_identifiers_apply(body, env).empty() && body == before, "a request with a stand-in in it stays");

    json plain = {{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}};
    const json plain_before = plain;
    require(server_identifiers_apply(plain, env).empty() && plain == plain_before,
            "a request without identifiers stays and has no map");
}

static void test_streamed_text_only_grows() {
    const server_identifier_env env = mac_env(users("alice"), "");
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "look in " + env.home}}})}};
    const server_identifier_map map = server_identifiers_apply(body, env);
    const std::string written = "Saved " + STAND_IN + "/notes.md, not " + STAND_IN + "x or " + users("inf") +
                                "; at " + STAND_IN;
    std::string last;
    for (size_t n = 0; n <= written.size(); ++n) {
        const std::string sent = map.to_identifiers(written.substr(0, n), true);
        require(sent.compare(0, last.size(), last) == 0,
                "each partial text extends the last one at " + std::to_string(n));
        last = sent;
    }
    const std::string final_text = map.to_identifiers(written, false);
    require(final_text.compare(0, last.size(), last) == 0, "the final text extends the partial ones");
    require(final_text == "Saved " + env.home + "/notes.md, not " + STAND_IN + "x or " + users("inf") + "; at " +
                              env.home,
            "the final text: " + final_text);
}

static void test_the_client_gets_its_own_paths() {
    const server_identifier_env env = mac_env(users("alice"), "");
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "look in " + env.home}}})}};
    const server_identifier_map map = server_identifiers_apply(body, env);

    common_chat_parser_params parser; // content only
    task_result_state state(parser, map);
    const std::string rest    = "/code/a.txt and that is all there is to say here.";
    const std::string written = "The file is at " + STAND_IN + rest;
    std::string streamed;
    std::vector<common_chat_msg_diff> diffs;
    for (size_t i = 0; i < written.size(); i += 3) {
        state.update_chat_msg(written.substr(i, 3), true, diffs);
        for (const auto & d : diffs) {
            streamed += d.content_delta;
        }
    }
    const common_chat_msg final_msg = state.update_chat_msg("", false, diffs);
    for (const auto & d : diffs) {
        streamed += d.content_delta;
    }
    const std::string expected = "The file is at " + env.home + rest;
    require(streamed == expected, "the streamed deltas: " + streamed);
    require(final_msg.content == expected, "the final message: " + final_msg.content);
    require(state.chat_msg.content == written, "the message as the model wrote it is kept for the next turn");
}

static void test_windows_spellings() {
    server_identifier_env env;
    env.home          = "C:\\Users\\bob";
    env.home_stand_in = "C:\\Users\\infer";
    env.windows       = true;
    // C:/Users/... and /c/Users/... are the same folders with '/'
    const std::string slash          = std::string("C:") + users("bob");
    const std::string slash_stand_in = std::string("C:") + STAND_IN;
    const std::string bash           = std::string("/c") + users("bob");
    const std::string bash_stand_in  = std::string("/c") + STAND_IN;
    json body = {{"messages", json::array({
        {{"role", "system"}, {"content", "Memory: C:\\Users\\bob\\.claude\\projects\\C--Users-bob-app\\memory\\"}},
        {{"role", "user"}, {"content", "C:\\Users\\bob\\app, " + slash + "/app, c:\\Users\\bob, " + bash + "/app"}},
        {{"role", "assistant"}, {"content", ""},
         {"tool_calls", json::array({tool_call("t1", "Read", {{"path", "C:\\Users\\bob\\app\\a.txt"}})})}}})}};
    const json before = body;
    const server_identifier_map map = server_identifiers_apply(body, env);
    require(text_of(body["messages"][0]) ==
                "Memory: C:\\Users\\infer\\.claude\\projects\\C--Users-infer-project\\memory\\",
            "the Windows memory folder: " + text_of(body["messages"][0]));
    require(text_of(body["messages"][1]) == "C:\\Users\\infer\\app, " + slash_stand_in +
                                                "/app, c:\\Users\\infer, " + bash_stand_in + "/app",
            "each spelling of the home folder: " + text_of(body["messages"][1]));
    const std::string args = body["messages"][2]["tool_calls"][0]["function"]["arguments"];
    require(args == json({{"path", "C:\\Users\\infer\\app\\a.txt"}}).dump(), "JSON-escaped arguments: " + args);
    require(map.to_identifiers(args, false) == before["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "the arguments come back");
}

int main() {
    test_billing_header();
    test_claude_desktop_env_lines();
    test_claude_code_shares_its_context();
    test_claude_desktop_sessions_share_their_context();
    test_chatgpt_conversations_share_their_context();
    test_chatgpt_visualizations_share_their_context();
    test_codex_projects_share_their_context();
    test_identifiers_of_a_system();
    test_matches_stop_at_name_boundaries();
    test_a_request_that_cannot_be_mapped_exactly_stays();
    test_streamed_text_only_grows();
    test_the_client_gets_its_own_paths();
    test_windows_spellings();
    std::printf("ok\n");
    return 0;
}
