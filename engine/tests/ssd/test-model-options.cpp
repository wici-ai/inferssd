// SPDX-License-Identifier: Apache-2.0
#include "ggml-ssd/settings/model-options.h"
#include <cassert>

int main() {
    using namespace ssd::settings;
    const model_identity a {"model-a-uuid", "deepseek41", "36", "2", "Test GPU", ""};
    const std::string rule = "fixture|on|model-a-uuid|deepseek41|36|2|Test GPU|1|0";
    auto configured = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(rule) + "]\n", "fixture");
    assert(configured);
    auto base = parse("", "");
    assert(base);
    assert(!has_model_rules(base.value));
    assert(has_model_rules(configured.value));
    auto defaults = resolve_model(base.value, a);
    assert(defaults && !defaults.value.engram_prefetch && defaults.value.hc_fused);
    const auto original = configured.value.json();
    auto selected = resolve_model(configured.value, a);
    assert(selected && selected.value.engram_prefetch && !selected.value.hc_fused);
    assert(selected.value.matched_rule == "fixture");
    assert(ssd::json::parse(selected.value.json(a)));
    for (int field = 0; field < 5; ++field) {
        auto b = a;
        auto * fact = field == 0 ? &b.uuid : field == 1 ? &b.architecture :
                     field == 2 ? &b.file_type : field == 3 ? &b.quantization_version : &b.hardware;
        *fact = "other";
        const auto unmatched = resolve_model(configured.value, b);
        assert(unmatched && !unmatched.value.engram_prefetch && unmatched.value.hc_fused);
        assert(unmatched.value.matched_rule.empty());
    }
    assert(resolve_model(configured.value, {}).value.matched_rule.empty());
    // Repeated and overlapping contexts do not modify the source or each other.
    for (int i = 0; i < 20; ++i) {
        assert(resolve_model(configured.value, a).value.engram_prefetch);
        assert(!resolve_model(configured.value, {}).value.engram_prefetch);
        assert(!resolve_model(base.value, a).value.engram_prefetch);
    }
    assert(configured.value.json().value == original.value);
    auto off_rule = rule; off_rule.replace(off_rule.find("|on|"), 4, "|off|");
    auto off = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(off_rule) + "]\n", "off");
    assert(off && !resolve_model(off.value, a).value.engram_prefetch);
    auto explicit_settings = parse("inferssd-settings 1\nmodel.engram_prefetch=1\nmodel.hc_fused=0\n", "legacy");
    assert(explicit_settings && resolve_model(explicit_settings.value, {}).value.engram_prefetch);
    assert(!resolve_model(explicit_settings.value, {}).value.hc_fused);
    auto cli = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(rule) + "]\n", "cli",
                     {{"model.engram_prefetch", "0"}, {"model.hc_fused", "1"}});
    assert(cli && !resolve_model(cli.value, a).value.engram_prefetch && resolve_model(cli.value, a).value.hc_fused);
    for (const auto & invalid : {"bad", "x|on|*|deepseek41|36|2|Test GPU|1|0", "x|yes|id|arch|36|2|GPU|1|0"}) {
        auto parsed = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(invalid) + "]\n", "invalid");
        assert(parsed && !resolve_model(parsed.value, a));
    }
    auto keep = parse("inferssd-settings 1\nmodel.engram_prefetch=1\nmodel.hc_fused=0\nmodel.optimizations=[" +
                      ssd::json::quote("keep|on|model-a-uuid|deepseek41|36|2|Test GPU|-|-") + "]\n", "keep");
    assert(keep && resolve_model(keep.value, a).value.engram_prefetch && !resolve_model(keep.value, a).value.hc_fused);
    for (const auto & literal : {"[" + ssd::json::quote(std::string(4097, 'x')) + "]",
                                "[" + ssd::json::quote(rule) + "," + ssd::json::quote(rule) + "]"}) {
        auto bounded = parse("inferssd-settings 1\nmodel.optimizations=" + literal + "\n", "bounded");
        assert(bounded && !resolve_model(bounded.value, a));
    }
    std::string many = "[";
    for (int i = 0; i < 65; ++i) { many += (i ? "," : "") + ssd::json::quote(std::to_string(i) + rule.substr(rule.find('|'))); }
    auto too_many = parse("inferssd-settings 1\nmodel.optimizations=" + many + "]\n", "bounded");
    assert(too_many && !resolve_model(too_many.value, a));
    auto ambiguous = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(rule) + "," +
                           ssd::json::quote("second" + rule.substr(rule.find('|'))) + "]\n", "ambiguous");
    assert(ambiguous && !resolve_model(ambiguous.value, a));

    // UUID-free models select existing options using loaded metadata only.
    const model_identity named {"", "deepseek41", "36", "2", "Test GPU", "Example Model"};
    const std::string named_rule = "named|on|name:Example Model|deepseek41|36|2|Test GPU|1|0";
    auto names = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(named_rule) + "]\n", "names");
    assert(names);
    const auto selected_name = resolve_model(names.value, named);
    assert(selected_name && selected_name.value.engram_prefetch && !selected_name.value.hc_fused);
    assert(selected_name.value.matched_rule == "named");
    auto with_uuid = named; with_uuid.uuid = "also-has-a-uuid";
    assert(resolve_model(names.value, with_uuid).value.engram_prefetch);
    for (int field = 0; field < 5; ++field) {
        auto changed = named;
        auto * fact = field == 0 ? &changed.name : field == 1 ? &changed.architecture :
                     field == 2 ? &changed.file_type : field == 3 ? &changed.quantization_version : &changed.hardware;
        *fact = "different";
        assert(resolve_model(names.value, changed).value.matched_rule.empty());
        *fact = "";
        assert(resolve_model(names.value, changed).value.matched_rule.empty());
    }
    auto uuid_rule = rule; uuid_rule.replace(uuid_rule.find("model-a-uuid"), 12, "uuid:model-a-uuid");
    auto uuid = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(uuid_rule) + "]\n", "uuid");
    assert(uuid && resolve_model(uuid.value, a).value.engram_prefetch);
    for (const auto & selector : {"name:", "uuid:", "name:*", "sha256:deadbeef"}) {
        auto invalid_rule = named_rule;
        invalid_rule.replace(invalid_rule.find("name:Example Model"), 18, selector);
        auto invalid = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(invalid_rule) + "]\n", "invalid-selector");
        assert(invalid && !resolve_model(invalid.value, named));
    }
    auto inactive = named_rule; inactive.replace(inactive.find("|on|"), 4, "|off|");
    auto inactive_settings = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(inactive) + "]\n", "off-name");
    assert(inactive_settings && !resolve_model(inactive_settings.value, named).value.engram_prefetch);
    auto name_cli = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(named_rule) + "]\n", "cli-name",
                          {{"model.engram_prefetch", "0"}, {"model.hc_fused", "1"}});
    assert(name_cli && !resolve_model(name_cli.value, named).value.engram_prefetch && resolve_model(name_cli.value, named).value.hc_fused);
}
