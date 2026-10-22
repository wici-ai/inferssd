#pragma once

#include "server-sampler-struct-top1.h"

#include <algorithm>

struct server_k3_protocol_ids {
    llama_token open = LLAMA_TOKEN_NULL, close = LLAMA_TOKEN_NULL;
    llama_token sep = LLAMA_TOKEN_NULL, end = LLAMA_TOKEN_NULL;

    bool valid() const { return open >= 0 && close >= 0 && sep >= 0 && end >= 0; }

    static server_k3_protocol_ids find(const std::vector<server_struct_token> & tokens) {
        server_k3_protocol_ids result;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (!tokens[i].special) { continue; }
            if (tokens[i].bytes == "<|open|>")       { result.open = i; }
            if (tokens[i].bytes == "<|close|>")      { result.close = i; }
            if (tokens[i].bytes == "<|sep|>")        { result.sep = i; }
            if (tokens[i].bytes == "<|end_of_msg|>") { result.end = i; }
        }
        return result;
    }
};

// Ordinary assistant-message profile. Body text, including JSON and quoted
// words "think/response/message", is opaque. Only actual special IDs start tags.
// When the request declares tools, the profile also admits the template's tool
// section after the response: tools > call > argument|json. Those openers carry
// attributes (ordinary bytes after the name and a space); values stay opaque.
// With tool_choice=required the response stays empty and the tool section is
// mandatory, which is what the tool-call grammar enforces for other templates.
class server_k3_protocol {
public:
    enum phase { INVALID, THINK_BODY, RESPONSE_BODY, AFTER_THINK, AFTER_RESPONSE,
                 AFTER_MESSAGE, OPEN_LABEL, CLOSE_LABEL, DONE,
                 AFTER_TOOLS, TOOLS_BODY, CALL_BODY, VALUE_BODY };

    void reset(const std::string & prefix, const std::vector<server_struct_token> & tokens,
               server_k3_protocol_ids ids, bool tags, bool force_response, bool tools = false,
               bool required = false) {
        *this = server_k3_protocol();
        tokens_ = &tokens;
        ids_ = ids;
        tags_ = tags;
        force_ = force_response;
        tools_ = tools;
        required_ = tools && required;
        const std::string message = "<|open|>message role=\"assistant\"<|sep|>";
        if (ids.valid() && prefix == message + "<|open|>think<|sep|>") {
            state_ = THINK_BODY;
            stack_ = {"message", "think"};
        } else if (ids.valid() && prefix == message + "<|open|>response<|sep|>") {
            state_ = RESPONSE_BODY;
            stack_ = {"message", "response"};
        } else if (tags || force_response) {
            throw std::invalid_argument("K3 protocol requires the ordinary assistant think/response generation prefix");
        }
    }

    bool observing() const { return state_ != INVALID; }
    phase state() const { return state_; }
    const std::vector<std::string> & stack() const { return stack_; }
    const std::string & label_bytes() const { return label_; }

    const char * name() const {
        switch (state_) {
            case THINK_BODY: return "think_body";
            case RESPONSE_BODY: return "response_body";
            case AFTER_THINK: return "after_think";
            case AFTER_RESPONSE: return "after_response";
            case AFTER_MESSAGE: return "after_message";
            case OPEN_LABEL: return "open_label";
            case CLOSE_LABEL: return "close_label";
            case DONE: return "done";
            case AFTER_TOOLS: return "after_tools";
            case TOOLS_BODY: return "tools_body";
            case CALL_BODY: return "call_body";
            case VALUE_BODY: return "value_body";
            default: return "unobserved";
        }
    }

    bool constrained() const {
        return tags_ || (force_ && (state_ == AFTER_THINK || (state_ == OPEN_LABEL && labels_ == response_label())));
    }

    // These states specify protocol bytes, not a choice between body and close.
    bool forcing() const {
        if (!constrained()) { return false; }
        switch (state_) {
            case OPEN_LABEL: return labels_.size() == 1 && !has_attributes(labels_[0]);
            case CLOSE_LABEL:
            case AFTER_MESSAGE:
            case AFTER_TOOLS: return true;
            case RESPONSE_BODY: return tags_ && required_;
            case AFTER_RESPONSE: return !tools_ || (tags_ && required_);
            case AFTER_THINK: return force_ || (tags_ && required_);
            default: return false;
        }
    }

    bool is_label_token(llama_token id) const {
        if ((state_ != OPEN_LABEL && state_ != CLOSE_LABEL) || !valid_id(id) || (*tokens_)[id].special) { return false; }
        const std::string next = label_ + (*tokens_)[id].bytes;
        if (next.size() == label_.size()) { return false; }
        static const std::vector<std::string> chat = {"think", "response", "message"};
        static const std::vector<std::string> with_tools = {"think", "response", "message", "tools", "call", "argument", "json"};
        for (const std::string & label : tools_ ? with_tools : chat) {
            if (next.size() <= label.size() && label.compare(0, next.size(), next) == 0) { return true; }
        }
        return false;
    }

    bool allows(llama_token id) const {
        if (!valid_id(id)) { return false; }
        if (!constrained()) { return true; }
        const bool required = tags_ && required_;
        switch (state_) {
            case THINK_BODY:
            case VALUE_BODY:
                return id == ids_.close || !(*tokens_)[id].special;
            case RESPONSE_BODY:
                // A required call leaves no answer text before it.
                return id == ids_.close || (!required && !(*tokens_)[id].special);
            case AFTER_THINK:
                // TAGS alone permits an omitted response; the separate FORCE
                // switch is what requires opening one after a natural close.
                return id == ids_.open || (!force_ && !required && id == ids_.close);
            case AFTER_RESPONSE: return (!required && id == ids_.close) || (tools_ && id == ids_.open);
            case AFTER_MESSAGE: return id == ids_.end;
            case AFTER_TOOLS: return id == ids_.close;
            case TOOLS_BODY: return id == ids_.open || (calls_ > 0 && id == ids_.close);
            case CALL_BODY: return id == ids_.close || (last_child_ != "json" && id == ids_.open);
            case OPEN_LABEL:
            case CLOSE_LABEL: {
                if (id == ids_.sep) { return !complete_label().empty(); }
                const auto & token = (*tokens_)[id];
                return !token.special && !token.bytes.empty() && label_allows(token.bytes);
            }
            default: return false;
        }
    }

    // Runs after budget/grammar masks and BEFORE any probability truncation.
    // Alternative tokenizations of the same label remain legal (byte prefixes).
    void mask(llama_token_data_array & candidates) const {
        bool any = false;
        for (size_t i = 0; i < candidates.size; ++i) {
            auto & token = candidates.data[i];
            if (!allows(token.id)) { token.logit = -INFINITY; }
            any |= std::isfinite(token.logit);
        }
        if (!any) { throw std::runtime_error("K3 constraint conflict: empty legal distribution before truncation"); }
    }

    // Call only for the final token, once. In observation-only mode a malformed
    // protocol stops label classification; it does not silently enforce TAGS.
    void accept(llama_token id) {
        if (constrained() && !allows(id)) { throw std::runtime_error("K3 constraint conflict: illegal final token"); }
        if (!observing() || !valid_id(id)) { return; }
        if (state_ == OPEN_LABEL || state_ == CLOSE_LABEL) {
            if (id == ids_.sep) {
                const std::string label = complete_label();
                if (label.empty()) { state_ = INVALID; return; }
                if (state_ == OPEN_LABEL) {
                    stack_.push_back(label);
                    state_ = label == "tools" ? TOOLS_BODY : label == "call" ? CALL_BODY :
                             label == "argument" || label == "json" ? VALUE_BODY : RESPONSE_BODY;
                    if (label == "call") { last_child_.clear(); }
                } else {
                    stack_.pop_back();
                    state_ = label == "think" ? AFTER_THINK : label == "response" ? AFTER_RESPONSE :
                             label == "tools" ? AFTER_TOOLS : label == "call" ? TOOLS_BODY :
                             label == "argument" || label == "json" ? CALL_BODY : AFTER_MESSAGE;
                    if (label == "call") { ++calls_; }
                    if (label == "argument" || label == "json") { last_child_ = label; }
                }
                label_.clear();
                labels_.clear();
            } else if ((*tokens_)[id].special || !label_allows((*tokens_)[id].bytes)) {
                state_ = INVALID;
            } else {
                label_ += (*tokens_)[id].bytes;
            }
            return;
        }
        if (id == ids_.close && (state_ == THINK_BODY || state_ == RESPONSE_BODY || state_ == VALUE_BODY ||
                                 state_ == AFTER_THINK || state_ == AFTER_RESPONSE || state_ == AFTER_TOOLS ||
                                 state_ == TOOLS_BODY || state_ == CALL_BODY)) {
            state_ = CLOSE_LABEL;
            labels_ = {stack_.back()};
            label_.clear();
        } else if (id == ids_.open && state_ == AFTER_THINK) {
            state_ = OPEN_LABEL;
            labels_ = response_label();
            label_.clear();
        } else if (id == ids_.open && ((tools_ && state_ == AFTER_RESPONSE) || state_ == TOOLS_BODY ||
                                       (state_ == CALL_BODY && last_child_ != "json"))) {
            labels_ = state_ == AFTER_RESPONSE ? std::vector<std::string>{"tools"} :
                      state_ == TOOLS_BODY     ? std::vector<std::string>{"call"} :
                      last_child_.empty()      ? std::vector<std::string>{"argument", "json"} :
                                                 std::vector<std::string>{"argument"};
            state_ = OPEN_LABEL;
            label_.clear();
        } else if (id == ids_.end && state_ == AFTER_MESSAGE) {
            state_ = DONE;
        } else if ((state_ != THINK_BODY && state_ != RESPONSE_BODY && state_ != VALUE_BODY) || (*tokens_)[id].special) {
            state_ = INVALID;
        }
    }

private:
    static const std::vector<std::string> & response_label() {
        static const std::vector<std::string> label = {"response"};
        return label;
    }
    static bool has_attributes(const std::string & label) {
        return label == "call" || label == "argument" || label == "json";
    }
    // Opening labels with attributes continue after "name " with ordinary bytes.
    bool label_allows(const std::string & bytes) const {
        const std::string next = label_ + bytes;
        for (const auto & label : labels_) {
            if (next.size() <= label.size() ? label.compare(0, next.size(), next) == 0 :
                    state_ == OPEN_LABEL && has_attributes(label) && next.compare(0, label.size() + 1, label + " ") == 0) {
                return true;
            }
        }
        return false;
    }
    // The label a separator would complete: a bare name, or an attributed opener.
    std::string complete_label() const {
        for (const auto & label : labels_) {
            const bool attributed = state_ == OPEN_LABEL && has_attributes(label);
            if (attributed ? label_.size() > label.size() + 1 && label_.compare(0, label.size() + 1, label + " ") == 0 :
                             label_ == label) {
                return label;
            }
        }
        return "";
    }
    bool valid_id(llama_token id) const { return tokens_ && id >= 0 && size_t(id) < tokens_->size(); }
    const std::vector<server_struct_token> * tokens_ = nullptr;
    server_k3_protocol_ids ids_;
    bool tags_ = false, force_ = false, tools_ = false, required_ = false;
    phase state_ = INVALID;
    std::vector<std::string> stack_;
    std::vector<std::string> labels_;
    std::string label_, last_child_;
    int calls_ = 0;
};
