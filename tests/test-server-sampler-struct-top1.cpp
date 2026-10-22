#include "server-sampler-struct-top1.h"
#include "server-k3-protocol.h"
#include "server-expert-guard.h"
#include "reasoning-budget.h"

#include <cassert>
#include <iostream>

static const std::string prefix = "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";

int main() {
    const std::vector<std::string> pieces = {
        "The", "}}", "{\"", "<|close|>", "Need", "\":", "text\n", "【内容】",
        "「」『』（）《》", "<|end_of_msg|>", "<|open|>", "<|sep|>", "think", "response",
        "thi", "nk", "message", " \"", "there's", "\"", "{\"a\":", "x < y", " \t", "EOS",
        std::string("\xff\n", 2), "\r\n", "“”", "中文", " ", "content",
    };
    server_sampler_struct_top1 policy;
    policy.config = server_sampler_struct_top1_config::parse("1", nullptr, nullptr);
    for (size_t i = 0; i < pieces.size(); ++i) {
        policy.tokens.push_back(server_struct_classify(pieces[i], i == 3 || i == 9 || i == 10 || i == 11, i == 9 || i == 23));
    }
    for (size_t i = policy.tokens.size(); i < 100; ++i) {
        policy.tokens.push_back(server_struct_classify("word" + std::to_string(i), false, false));
    }
    for (int i : {1, 2, 3, 8, 9, 10, 11, 17, 19, 23, 25, 26}) { assert(policy.is_struct(i)); }
    for (int i : {0, 4, 5, 6, 7, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24, 27, 28}) { assert(!policy.is_struct(i)); }
    for (int i : {5, 6, 7, 18, 20, 21, 24}) { assert(policy.tokens[i].mixed); }
    assert(server_struct_bytes_hex(policy.tokens[24].bytes) == "ff0a");
    assert(!policy.is_struct(-1) && !policy.is_struct(100));
    auto tail = [&](int sampled, bool forced = false, bool label = false) {
        llama_token_data data[] = {{0, 0, .75f}, {sampled, 0, .25f}};
        llama_token_data_array q = {data, 2, 1, true};
        return policy.apply(sampled, q, forced, label);
    };
    assert(tail(1).replaced && tail(1).token == 0);
    assert(tail(1).sampled_p == .25 && tail(1).top1_p == .75);
    assert(!tail(4).replaced);
    for (int i : {5, 6, 7, 18, 20, 21, 24}) { assert(!tail(i).replaced); }
    assert(tail(17).replaced && tail(19).replaced); // pure quote, including leading whitespace
    assert(!tail(1, true).replaced); // budget/protocol-forced exemption
    assert(!tail(12).replaced && tail(12, false, true).replaced); // label only in label position
    policy.config.enabled = false;
    assert(!tail(1).replaced);
    policy.config.enabled = true;
    policy.config.min_top1_p = .75;
    assert(!tail(1).replaced); // strict >, not >=
    policy.config.min_top1_p = .5;
    assert(tail(1).replaced);
    policy.config.min_top1_p = 0;
    {
        llama_token_data data[] = {{1, 0, .75f}, {0, 0, .25f}};
        llama_token_data_array q = {data, 2, 0, true};
        assert(!policy.apply(1, q).replaced); // structural effective top-1 stays
    }
    assert(!server_sampler_struct_top1_config::parse(nullptr, nullptr, nullptr).active());
    assert(server_sampler_struct_top1_config::parse("0", nullptr, nullptr, "1").active());
    assert(server_sampler_struct_top1_config::parse("0", nullptr, nullptr, "0", "1").active());
    for (const char * value : {"-0.1", "1.1", "nan", "inf", "", "0.2junk"}) {
        bool rejected = false;
        try { server_sampler_struct_top1_config::parse("1", nullptr, value); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    policy.config = server_sampler_struct_top1_config::parse("1", "quotes,newline", "0");
    assert(!policy.is_struct(1) && policy.is_struct(2) && !policy.is_struct(6));
    policy.config = server_sampler_struct_top1_config::parse("1", nullptr, nullptr);

    // Actual penalties/temp/top-k/top-p/dist: raw top-1 is The, but repetition
    // penalty makes }} the effective top-1. q, not raw logits, decides.
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(8, 10, 0, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(2));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(.7f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(.95f, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(1));
    llama_sampler_accept(chain, 0);
    llama_token_data transformed[] = {{0, 4, 0}, {1, 3, 0}, {4, -10, 0}};
    llama_token_data_array q = {transformed, 3, -1, false};
    llama_sampler_apply(chain, &q);
    assert(q.data[q.selected].id == 1);
    assert(!policy.apply(1, q).replaced && policy.apply(1, q).top1 == 1);
    llama_sampler_free(chain);

    // The projection neither samples again nor consumes RNG: subsequent draws
    // remain identical to a clone given the same logits and one draw per step.
    auto * dist = llama_sampler_init_dist(1);
    auto * reference = llama_sampler_clone(dist);
    int replacements = 0;
    for (int i = 0; i < 100; ++i) {
        llama_token_data a[] = {{0, std::log(.75f), 0}, {1, std::log(.25f), 0}};
        llama_token_data b[] = {{0, std::log(.75f), 0}, {1, std::log(.25f), 0}};
        llama_token_data_array qa = {a, 2, -1, false}, qb = {b, 2, -1, false};
        llama_sampler_apply(dist, &qa);
        llama_sampler_apply(reference, &qb);
        assert(a[qa.selected].id == b[qb.selected].id);
        replacements += policy.apply(a[qa.selected].id, qa).replaced;
    }
    assert(replacements > 0);
    llama_sampler_free(dist);
    llama_sampler_free(reference);

    const auto ids = server_k3_protocol_ids::find(policy.tokens);
    assert(ids.valid());
    auto protocol = [&](bool tags, bool force) {
        server_k3_protocol result;
        result.reset(prefix, policy.tokens, ids, tags, force);
        return result;
    };
    auto p = protocol(true, false);
    assert(p.allows(3) && !p.allows(9) && !p.allows(10)); // empty think is legal
    assert(!p.is_label_token(12) && p.allows(12));
    // Record the known cost: unforced legal empty think close may become body.
    const auto empty_think = tail(3);
    assert(empty_think.replaced);
    p.accept(empty_think.token);
    assert(p.state() == server_k3_protocol::THINK_BODY);
    std::cout << "known_cost: legal empty-think close -> body effective top-1\n";
    p.accept(3);
    assert(p.is_label_token(12) && p.is_label_token(13));
    assert(p.allows(12) && !p.allows(13) && !p.allows(11));
    assert(p.forcing() && !tail(12, p.forcing(), true).replaced);
    p.accept(14); p.accept(15); // multi-token thi + nk
    assert(p.allows(11) && !p.allows(12));
    p.accept(11);
    assert(p.state() == server_k3_protocol::AFTER_THINK && p.allows(3) && p.allows(10));
    p.accept(10); p.accept(13); p.accept(11);
    assert(p.state() == server_k3_protocol::RESPONSE_BODY);
    // Nested JSON / quotation tokens are body data, never protocol stack edits.
    for (int i : {2, 20, 2, 19, 12, 19, 1, 25, 18}) { assert(p.allows(i)); p.accept(i); }
    assert(p.stack() == std::vector<std::string>({"message", "response"}));
    p.accept(3); assert(!p.allows(12)); p.accept(13); p.accept(11);
    assert(p.forcing() && !p.allows(9));
    p.accept(3); p.accept(16); p.accept(11); assert(p.allows(9) && !p.allows(23));
    p.accept(9); assert(p.state() == server_k3_protocol::DONE);
    for (bool tags : {false, true}) {
        p = protocol(tags, true);
        p.accept(3); p.accept(12); p.accept(11);
        assert(p.forcing() && p.allows(10) && !p.allows(3));
        p.accept(10); p.accept(13); p.accept(11);
        assert(!p.forcing() && p.state() == server_k3_protocol::RESPONSE_BODY);
    }
    p = protocol(true, false); p.accept(3);
    // Legal label originally below top-40: mask must rescue it BEFORE top-k.
    std::vector<llama_token_data> candidates;
    for (size_t i = 0; i < policy.tokens.size(); ++i) { candidates.push_back({(llama_token)i, i >= 30 ? 10.0f : -20.0f, 0}); }
    q = {candidates.data(), candidates.size(), -1, false};
    p.mask(q);
    auto * topk = llama_sampler_init_top_k(40);
    auto * sample = llama_sampler_init_dist(1);
    llama_sampler_apply(topk, &q); llama_sampler_apply(sample, &q);
    assert(p.allows(q.data[q.selected].id));
    llama_sampler_free(topk); llama_sampler_free(sample);
    // Simulate a conflicting existing grammar/budget mask. Fail, no relaxation.
    for (auto & t : candidates) { t.logit = t.id == 13 ? 0 : -INFINITY; }
    q = {candidates.data(), candidates.size(), -1, false};
    bool conflict = false;
    try { p.mask(q); } catch (const std::runtime_error &) { conflict = true; }
    assert(conflict);

    // Normal forced budget and protocol FORCE_RESPONSE agree, including the
    // overlapping close-think/open-response suffix; accept each final once.
    const std::vector<llama_token> start = {10, 12, 11}, end = {3, 12, 11};
    const std::vector<llama_token> forced = {3, 12, 11, 10, 13, 11};
    auto * budget = common_reasoning_budget_init(nullptr, start, end, forced, 0, REASONING_BUDGET_COUNTING);
    p = protocol(true, true);
    for (auto id : forced) {
        candidates.clear();
        for (size_t i = 0; i < policy.tokens.size(); ++i) { candidates.push_back({(llama_token)i, 0, 0}); }
        q = {candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(budget, &q); p.mask(q);
        auto * one = llama_sampler_init_dist(1); llama_sampler_apply(one, &q);
        const auto result = policy.apply(q.data[q.selected].id, q, true);
        assert(result.token == id && !result.replaced);
        llama_sampler_accept(budget, id); p.accept(id); llama_sampler_free(one);
    }
    assert(common_reasoning_budget_get_state(budget) == REASONING_BUDGET_DONE);
    assert(p.state() == server_k3_protocol::RESPONSE_BODY);
    llama_sampler_free(budget);
    // Budget expiry on an already-started natural close must not inject a
    // duplicate close into the tag. This behavior is opt-in; old mode remains.
    for (int budget_size : {1, 2}) {
        budget = common_reasoning_budget_init(nullptr, start, end, forced, budget_size, REASONING_BUDGET_COUNTING);
        const std::vector<llama_token> alternative_end = {3, 14, 15, 11};
        for (auto id : alternative_end) {
            common_reasoning_budget_set_protocol_end(budget, true, id == 11);
            llama_sampler_accept(budget, id);
            assert(common_reasoning_budget_get_state(budget) != REASONING_BUDGET_FORCING);
        }
        assert(common_reasoning_budget_get_state(budget) == REASONING_BUDGET_DONE);
        llama_sampler_free(budget);
    }
    // Declared tools: the template's tool section after the response.
    {
        const std::vector<std::string> tool_pieces = {
            "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>", "think", "response", "message", "tools", "call",
            " tool=\"web_search\" index=\"1\"", "argument", " key=\"query\" type=\"string\"", "json", " type=\"object\"",
            "Kimi K3", "<|close|>argument<|sep|>", "{\"query\":1}", "ca", "ll", "calls", "Reason.",
        };
        enum { OPEN, CLOSE, SEP, END, THINK, RESPONSE, MESSAGE, TOOLS, CALL, CALL_ATTRS, ARGUMENT, ARG_ATTRS,
               JSON, JSON_ATTRS, VALUE, LITERAL_CLOSE, JSON_VALUE, CA, LL, CALLS, REASON };
        std::vector<server_struct_token> tool_tokens;
        for (size_t i = 0; i < tool_pieces.size(); ++i) {
            tool_tokens.push_back(server_struct_classify(tool_pieces[i], i <= END, i == END));
        }
        const auto tool_ids = server_k3_protocol_ids::find(tool_tokens);
        assert(tool_ids.valid());
        auto tool_protocol = [&](bool tags, bool force, bool tools) {
            server_k3_protocol result;
            result.reset(prefix, tool_tokens, tool_ids, tags, force, tools);
            return result;
        };
        auto run = [](server_k3_protocol & protocol, std::initializer_list<int> ids) {
            for (int id : ids) { assert(protocol.allows(id)); protocol.accept(id); }
        };
        auto t = tool_protocol(true, true, true);
        run(t, {REASON, CLOSE, THINK, SEP, OPEN, RESPONSE, SEP, CLOSE, RESPONSE, SEP});
        assert(t.state() == server_k3_protocol::AFTER_RESPONSE && !t.forcing());
        assert(t.allows(OPEN) && t.allows(CLOSE) && !t.allows(END) && !t.allows(VALUE));
        run(t, {OPEN});
        assert(t.allows(TOOLS) && !t.allows(CALL) && !t.allows(SEP));
        run(t, {TOOLS, SEP});
        assert(t.state() == server_k3_protocol::TOOLS_BODY && !t.allows(CLOSE) && !t.allows(VALUE)); // at least one call
        run(t, {OPEN});
        assert(!t.forcing() && t.is_label_token(CA) && !t.allows(ARGUMENT));
        run(t, {CA});
        assert(!t.allows(SEP) && !t.allows(CALLS) && t.allows(LL));
        run(t, {LL});
        assert(!t.allows(SEP) && t.allows(CALL_ATTRS)); // a call names its tool
        run(t, {CALL_ATTRS});
        assert(t.allows(SEP) && t.allows(VALUE));
        run(t, {SEP});
        assert(t.state() == server_k3_protocol::CALL_BODY && t.allows(OPEN) && t.allows(CLOSE));
        run(t, {OPEN, ARGUMENT, ARG_ATTRS, SEP, VALUE, LITERAL_CLOSE});
        assert(t.state() == server_k3_protocol::VALUE_BODY);
        assert(t.stack() == std::vector<std::string>({"message", "tools", "call", "argument"}));
        run(t, {CLOSE, ARGUMENT, SEP});
        {
            auto next = t;
            run(next, {OPEN});
            assert(next.allows(ARGUMENT) && !next.allows(JSON)); // no JSON block after tagged arguments
        }
        run(t, {CLOSE, CALL, SEP});
        assert(t.state() == server_k3_protocol::TOOLS_BODY && t.allows(CLOSE) && t.allows(OPEN));
        run(t, {CLOSE, TOOLS, SEP});
        assert(t.state() == server_k3_protocol::AFTER_TOOLS && t.forcing() && !t.allows(OPEN));
        run(t, {CLOSE, MESSAGE, SEP, END});
        assert(t.state() == server_k3_protocol::DONE);

        t = tool_protocol(true, true, true);
        run(t, {CLOSE, THINK, SEP, OPEN, RESPONSE, SEP, CLOSE, RESPONSE, SEP, OPEN, TOOLS, SEP,
                OPEN, CALL, CALL_ATTRS, SEP, OPEN, JSON, JSON_ATTRS, SEP, JSON_VALUE, CLOSE, JSON, SEP});
        assert(t.state() == server_k3_protocol::CALL_BODY && !t.allows(OPEN) && t.allows(CLOSE));

        // Without declared tools the ordinary profile is unchanged: the message must close.
        t = tool_protocol(true, true, false);
        run(t, {CLOSE, THINK, SEP, OPEN, RESPONSE, SEP, CLOSE, RESPONSE, SEP});
        assert(t.forcing() && !t.allows(OPEN) && t.allows(CLOSE) && !t.is_label_token(CA));

        // FORCE_RESPONSE alone constrains only the response opener, not tool labels.
        t = tool_protocol(false, true, true);
        run(t, {CLOSE, THINK, SEP, OPEN, RESPONSE, SEP, CLOSE, RESPONSE, SEP, OPEN});
        assert(!t.constrained() && t.allows(VALUE));
        run(t, {TOOLS, SEP});
        assert(t.state() == server_k3_protocol::TOOLS_BODY);

        // tool_choice=required: no answer text, and the tool section must follow.
        for (bool force : {false, true}) {
            t.reset(prefix, tool_tokens, tool_ids, true, force, true, true);
            run(t, {REASON, CLOSE, THINK, SEP});
            assert(t.state() == server_k3_protocol::AFTER_THINK && t.forcing() && t.allows(OPEN) && !t.allows(CLOSE));
            run(t, {OPEN, RESPONSE, SEP});
            assert(t.forcing() && t.allows(CLOSE) && !t.allows(VALUE));
            run(t, {CLOSE, RESPONSE, SEP});
            assert(t.forcing() && t.allows(OPEN) && !t.allows(CLOSE));
            run(t, {OPEN, TOOLS, SEP, OPEN, CALL, CALL_ATTRS, SEP, OPEN, ARGUMENT, ARG_ATTRS, SEP, VALUE, CLOSE, ARGUMENT,
                    SEP, CLOSE, CALL, SEP, CLOSE, TOOLS, SEP, CLOSE, MESSAGE, SEP, END});
            assert(t.state() == server_k3_protocol::DONE);
        }
        // Required without declared tools leaves the ordinary profile unchanged.
        t.reset(prefix, tool_tokens, tool_ids, true, true, false, true);
        run(t, {CLOSE, THINK, SEP, OPEN, RESPONSE, SEP});
        assert(t.allows(VALUE) && !t.forcing());
    }
    server_expert_guard guard;
    guard.reset(0, {}, 0, 3, 10, 3);
    guard.accept_generated(tail(3).token);
    assert(guard.control_triggers() == 0);
    guard.accept_generated(3);
    assert(guard.control_triggers() == 1);
    std::cout << "raw bytes, pure/mixed classes, effective q, MIN_TOP1P, RNG, protocol ordering, pre-truncation masks, conflicts, budget and final-token guard: passed\n";
}
