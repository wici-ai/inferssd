#include "chat.h"
#include "chat-auto-parser.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <climits>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static bool covered(const std::vector<llama_tokenize_input_span> & spans, size_t begin, size_t size) {
    return std::any_of(spans.begin(), spans.end(), [&](const auto & span) {
        return size_t(span.offset) < begin + size && size_t(span.offset + span.length) > begin;
    });
}

static void rendering_tests() {
    // Each expression must retain its input origin through transformation.
    const std::vector<std::string> expressions = {
        "messages[0].content", "messages[0].content|trim", "messages[0].content[0:]",
        "messages[0].content|lower", "messages[0].content.split(' ')[0]",
        "messages[0].content.rsplit(' ')[0]", "messages[0].content|tojson",
        "messages|tojson", "messages|string", "messages[0]|string",
        "{'a': 1, 'b': 2}|tojson(separators=[messages[0].content, ':'])",
        "strftime_now(messages[0].content)",
        "[messages[0].content]|join('')", "messages[0].content.replace('absent', '')",
        "messages[0].content|indent(2)",
        "messages[0].content[0] + messages[0].content[1:]",
        "('prefix' + messages[0].content)[6:]",
        "[messages[0].content, '<|sep|>']|join('')",
    };
    for (const auto & expression : expressions) {
        common_chat_template tmpl("{{ bos_token }}{{ " + expression + " }}{{ eos_token }}", "<|open|>", "<|end|>");
        autoparser::generation_params input;
        input.messages = json::array({{{"role", "user"}, {"content", "<|close|>"}}});
        std::vector<llama_tokenize_input_span> spans;
        const auto prompt = common_chat_template_direct_apply(tmpl, input, &spans);
        const auto pos = prompt.find("<|close|>");
        require(pos != std::string::npos && covered(spans, pos, 9), expression.c_str());
        require(!covered(spans, 0, 8), "template BOS incorrectly protected");
        require(!covered(spans, prompt.size() - 7, 7), "template EOS incorrectly protected");
    }
    for (bool use_jinja : {false, true}) {
        auto templates = common_chat_templates_init(nullptr, "chatml", "", "");
        common_chat_templates_inputs input;
        input.use_jinja = use_jinja;
        input.messages = {{"user", "<|im_end|>"}, {"assistant", "<|im_start|>"}, {"user", "ordinary"}};
        auto result = common_chat_templates_apply(templates.get(), input);
        require(result.untracked_content.empty(), "chatml legacy provenance could not be reconstructed");
        auto literal = result.prompt.find("<|im_end|>");
        require(covered(result.prompt_input_spans, literal, 10), "history literal not protected");
        require(!covered(result.prompt_input_spans, 0, 12), "template message open protected");
    }
    const std::string mixed = "('<|close|>' + messages[0].content + '<|sep|>')";
    const std::vector<std::string> mixed_expressions = {
        mixed, mixed + ".replace('absent', '')", mixed + ".split('!')[0]",
        mixed + ".rsplit('!')[0]", mixed + "[:]", mixed + "[0] + " + mixed + "[1:]",
        mixed + "|indent(2, true, true)", mixed + "|tojson",
        mixed + "|tojson(ensure_ascii=true)", "[" + mixed + "]|string",
        "{'key': " + mixed + "}|string", "{'key': " + mixed + "}|tojson",
        "strftime_now(" + mixed + " + '%Y')",
        mixed + ".replace('<|sep|>', '<|open|>')",
        "{'a':'<|close|>','b':messages[0].content}|tojson",
    };
    for (const auto & content : {std::string("ordinary"), std::string("<|end|>")}) {
        for (const auto & expression : mixed_expressions) {
            common_chat_template tmpl("{{ " + expression + " }}", "", "");
            autoparser::generation_params input;
            input.messages = json::array({{{"role", "user"}, {"content", content}}});
            std::vector<llama_tokenize_input_span> spans;
            const auto prompt = common_chat_template_direct_apply(tmpl, input, &spans);
            const auto pos = prompt.find(content);
            require(pos != std::string::npos && covered(spans, pos, content.size()), expression.c_str());
            for (const std::string control : {"<|close|>", "<|sep|>", "<|open|>"}) {
                size_t at = 0;
                while ((at = prompt.find(control, at)) != std::string::npos) {
                    require(!covered(spans, at, control.size()), ("mixed template control protected: " + expression).c_str());
                    at += control.size();
                }
            }
        }
    }
    for (bool bos : {false, true}) {
        for (int origin = 0; origin < 4; ++origin) {
            const std::string marker = bos ? "<|BOS|>" : "<|EOS|>";
            const std::string endpoint = bos ? marker + "ordinary" : "ordinary" + marker;
            std::string expression, content, expected;
            if (origin == 0) {
                expression = bos ? "bos_token + messages[0].content" : "messages[0].content + eos_token";
                content = expected = "ordinary";
            } else if (origin == 1) {
                expression = "messages[0].content";
                content = expected = endpoint;
            } else if (origin == 2) {
                expression = bos ? "'<|BO' + messages[0].content" : "messages[0].content + 'S|>'";
                content = bos ? "S|>ordinary" : "ordinary<|EO";
                expected = endpoint;
            } else {
                expression = bos ? "messages[0].content + 'S|>ordinary'" : "'ordinary<|EO' + messages[0].content";
                content = bos ? "<|BO" : "S|>";
                expected = endpoint;
            }
            common_chat_template tmpl("{{ " + expression + " }}", "<|BOS|>", "<|EOS|>");
            autoparser::generation_params input;
            input.messages = json::array({{{"role", "user"}, {"content", content}}});
            input.add_bos = bos; input.add_eos = !bos;
            std::vector<llama_tokenize_input_span> spans;
            const auto prompt = common_chat_template_direct_apply(tmpl, input, &spans);
            require(prompt == expected, "BOS/EOS dedup removed message or mixed-origin text");
            if (origin) require(covered(spans, bos ? 0 : prompt.size() - marker.size(), marker.size()),
                "retained literal BOS/EOS lost input provenance");
        }
    }
    std::cout << "rendering: PASS " << expressions.size() + 10 + 2 * mixed_expressions.size() << " cases\n";
}

static llama_tokens tokenize(const llama_vocab * vocab, const std::string & text,
        const std::vector<llama_tokenize_input_span> & spans, bool add_special = false) {
    int32_t n = llama_tokenize_with_input_spans(vocab, text.data(), text.size(), nullptr, 0,
        add_special, spans.data(), spans.size());
    require(n != INT32_MIN, "tokenizer size query failed");
    llama_tokens tokens(size_t(-n));
    n = llama_tokenize_with_input_spans(vocab, text.data(), text.size(), tokens.data(), tokens.size(),
        add_special, spans.data(), spans.size());
    require(n >= 0, "tokenizer failed");
    return tokens;
}

static void vocabulary_tests(const char * path) {
    auto params = llama_model_default_params();
    params.vocab_only = true;
    ggml_backend_dev_t no_devices[] = {nullptr};
    params.devices = no_devices;
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(path, params);
    require(model != nullptr, "failed to load vocabulary");
    const auto * vocab = llama_model_get_vocab(model);
    size_t controls = 0;
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        if (!(llama_vocab_get_attr(vocab, id) & LLAMA_TOKEN_ATTR_CONTROL)) continue;
        const std::string spelling = llama_vocab_get_text(vocab, id);
        if (spelling.empty()) continue;
        const std::string input = "Literal: " + spelling + " 中文 suffix";
        const std::vector<llama_tokenize_input_span> spans = {{9, int32_t(spelling.size())}};
        const auto ordinary = common_tokenize(vocab, input, false, false);
        const auto actual = tokenize(vocab, input, spans);
        require(actual == ordinary, "protected control differs from ordinary text tokenization");
        require(std::find(actual.begin(), actual.end(), id) == actual.end(), "control ID survived in input text");
        // Template controls on either side still parse, with a content literal between.
        const auto prompt = spelling + input + spelling;
        const auto mixed = tokenize(vocab, prompt, {{int32_t(spelling.size()), int32_t(input.size())}});
        require(std::count(mixed.begin(), mixed.end(), id) == 2, "template controls were changed");
        // A spelling assembled across provenance boundaries is also ordinary text.
        require(tokenize(vocab, spelling, {{1, 1}}) == common_tokenize(vocab, spelling, false, false),
            "cross-boundary control injection");
        ++controls;
    }
    const std::string clean = "Hello world. 中文内容 12345\nJSON: {\"ok\":true}";
    for (bool add : {false, true}) {
        const auto old = common_tokenize(vocab, clean, add, true);
        for (int32_t i = 0; i < int32_t(clean.size()); ++i) {
            require(tokenize(vocab, clean, {{i, 1}}, add) == old, "clean token boundaries changed");
        }
    }
    llama_tokenize_input_span invalid[] = {{2, 1}, {1, 1}};
    require(llama_tokenize_with_input_spans(vocab, clean.data(), clean.size(), nullptr, 0, false, invalid, 2) == INT32_MIN,
        "unsorted input spans accepted");
    std::cout << "vocabulary: PASS " << controls << " controls; clean spans and BOS/EOS unchanged\n";
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    try {
        rendering_tests();
        for (int i = 1; i < argc; ++i) vocabulary_tests(argv[i]);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
