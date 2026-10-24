#include "server-common.h"
#include "llama-cpp.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

static void check_vocabulary(const char * path) {
    auto params = llama_model_default_params();
    ggml_backend_dev_t no_devices[] = { nullptr };
    params.devices = no_devices; // Avoid even GPU enumeration in a CUDA build.
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    llama_model_ptr model(llama_model_load_from_file(path, params));
    require(bool(model), "vocabulary load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    std::string control;
    llama_token control_id = LLAMA_TOKEN_NULL;
    for (const std::string spelling : {"<|close|>", "<|im_end|>"}) {
        const auto ids = common_tokenize(vocab, spelling, false, true);
        if (ids.size() == 1 && llama_vocab_is_control(vocab, ids[0])) {
            control = spelling; control_id = ids[0]; break;
        }
    }
    require(control_id != LLAMA_TOKEN_NULL, "expected a Kimi or Qwen2 control vocabulary");
    const std::string literal = "literal A " + control + " B";
    auto message = [](const std::string & role, const json & content) {
        return json{{"role", role}, {"content", content}};
    };
    struct testcase { std::string name; json messages; bool prefill; bool literal; };
    const std::vector<testcase> cases = {
        {"ordinary", json::array({message("user", "literal A ordinary B")}), false, false},
        {"literal", json::array({message("user", literal)}), false, true},
        {"repeated", json::array({message("user", control + control)}), false, true},
        {"utf8", json::array({message("user", "中文 μ " + literal)}), false, true},
        {"text-parts", json::array({message("user", json::array({
            {{"type", "text"}, {"text", "A " + control}},
            {{"type", "text"}, {"text", " B"}}}))}), false, true},
        {"history", json::array({message("user", literal), message("assistant", "ordinary"),
                                 message("user", literal)}), false, true},
        {"system", json::array({message("system", literal), message("user", "ordinary")}), false, true},
        {"assistant-prefill", json::array({message("user", "ordinary"),
                                           message("assistant", literal)}), true, true},
    };
    const std::string text = "{% for message in messages %}{{ '" + control +
        "' }}{{ message['content'] }}{{ '" + control + "' }}{% endfor %}";
    size_t checked = 0;
    for (const auto & item : cases) {
        server_chat_params options{};
        options.use_jinja = true;
        options.enable_thinking = false;
        options.prefill_assistant = item.prefill;
        options.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        options.tmpls = common_chat_templates_init(model.get(), text);
        // Caller-supplied internal fields must not replace server-derived provenance.
        json body = {{"messages", item.messages}, {"chat_input_spans", json::array()},
                     {"prompt", "ignored caller prompt"}, {"parse_special", true}};
        std::vector<raw_buffer> files;
        const auto parsed = oaicompat_chat_params_parse(body, options, files);
        const auto prompt = parsed.at("prompt").get<std::string>();
        require(prompt != "ignored caller prompt", item.name + ": caller replaced rendered prompt");
        const auto actual = tokenize_chat_prompt(vocab, parsed);
        const auto old = common_tokenize(vocab, prompt, true, true);
        const size_t template_messages = item.messages.size() - (item.prefill ? 1 : 0);
        const auto expected_controls = int64_t(2 * template_messages);
        require(std::count(actual.begin(), actual.end(), control_id) == expected_controls,
                item.name + ": content control survived or a template control disappeared");
        require(!parsed.at("chat_input_spans").empty(), item.name + ": missing input provenance");
        if (item.literal) {
            require(std::count(old.begin(), old.end(), control_id) > expected_controls,
                    item.name + ": fixture did not reproduce the old interpretation");
        } else {
            require(actual == old, "ordinary token IDs changed");
        }
        ++checked;
    }
    const auto explicit_raw = tokenize_mixed(vocab, json(literal), false, true);
    require(explicit_raw == common_tokenize(vocab, literal, false, true), "raw tokenizer contract changed");
    require(std::count(explicit_raw.begin(), explicit_raw.end(), control_id) == 1,
            "explicit raw special parsing stopped working");
    std::cout << "SERVER_CONTROL_TOKEN_DELIVERY_PASS cases=" << checked
              << " raw_contract=1 vocab_only=1 model_evaluations=0\n";
}

int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    try {
        for (int i = 1; i < argc; ++i) check_vocabulary(argv[i]);
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
