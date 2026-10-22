#include "kimi-cpu-cache-policy.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
static void require(bool ok, const char * why) { if (!ok) { throw std::runtime_error(why); } }
int main() {
 try {
    unsetenv("GGML_KIMI_CPU_SINK_CACHE_COMPUTE");
    kimi_cache_compute_config c{}; char error[256] = {};
    require(kimi_cache_compute_config_from_env(&c, error, sizeof(error)), error);
    require(c.mode == 0 && c.apply_guard == 0, "default off and guard bypass");
    const int32_t ids[] = {40,10,70,20,80,60};
    const float w[] = {.40f,.20f,.15f,.10f,.08f,.07f};
    const int32_t sid[] = {70,40,10,20};
    const float sw[] = {.15f/.85f,.40f/.85f,.20f/.85f,.10f/.85f};
    kimi_cache_compute_plan p{};
    auto init = [&](uint32_t cache, bool guard = false) {
        require(kimi_cache_compute_plan_init(&c,ids,w,6,sid,sw,4,cache,guard,&p,error,sizeof(error)), error);
    };
    c.mode = KIMI_CACHE_COMPUTE_BONUS; init(2|16|32);
    require(p.base_mask == 15 && p.bonus_mask == 16 && p.final_mask == 31 && p.pin_mask == (2|16), "bonus selection/pins");
    c.mode = KIMI_CACHE_COMPUTE_HYBRID; init(2|16|32);
    require(p.protected_mask == 7 && p.pin_mask == (2|16|32), "protected mass and replacement pin pool");
    kimi_cache_compute_plan_update(&p, 0, 999); require(p.final_mask == 31 && !p.deadline_applied, "no early deadline");
    kimi_cache_compute_plan_update(&p, 0, 1000);
    require(p.dropped_mask == 8 && p.replacement_mask == 16 && p.bonus_mask == 32 && p.final_mask == 55, "hybrid plus one");
    const auto fixed = p; kimi_cache_compute_plan_update(&p, 63, 1001);
    require(!std::memcmp(&fixed,&p,sizeof(p)), "late arrivals cannot rejoin");
    init(2|16|32); kimi_cache_compute_plan_update(&p, 8, 1000);
    require(p.dropped_mask == 0 && p.final_mask == 31, "arrival at deadline retained");
    c.mode = KIMI_CACHE_COMPUTE_SHRINK; c.s_fraction = .5f; init(2|4|16|32);
    require(p.base_mask == 3 && p.bonus_mask == 4 && p.final_mask == 7, "shrink then reuse pruned cached S");
    init(63,true); require(p.config.mode == 0 && p.final_mask == 15 && p.bonus_mask == 0, "guard bypass");
    c.apply_guard = 1; init(63,true); require(p.base_mask == 3 && p.final_mask == 7, "explicit apply guard");
    c.apply_guard = 0;
    for (int mode=0; mode<=3; ++mode) for (uint32_t cache=0;cache<64;++cache) for (uint32_t ready=0;ready<64;++ready) {
        c.mode=mode; init(cache); kimi_cache_compute_plan_update(&p,ready,1000);
        require(!(p.final_mask & ~63u), "outside native pool");
        require(!((p.final_mask & ~p.base_mask) & ~cache), "uncached extra selected");
        require(!(p.protected_mask & ~p.final_mask), "protected expert dropped");
        require(!((p.final_mask & ~p.base_mask) & ~p.pin_mask), "extra without pin");
        require(!(p.replacement_mask & p.bonus_mask), "double counted replacement/bonus");
    }
    const int32_t bad[] = {40,10,70,20,80,80};
    require(!kimi_cache_compute_plan_init(&c,bad,w,6,sid,sw,4,0,false,&p,error,sizeof(error)), "duplicate native accepted");
    std::puts("CACHE_COMPUTE_POLICY_PASS explicit deadline/guard/shrink cases and 16384 cache/ready states"); return 0;
 } catch (const std::exception & e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
