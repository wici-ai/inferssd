static bool expert_pack_env_bool(const char *name, bool default_value) {
    const char *env = std::getenv(name);
    if (!env || !env[0]) return default_value;
    return env[0] != '0';
}

static void expert_pack_append_env_list(std::vector<std::string> &paths, const char *env_name) {
    const char *env = std::getenv(env_name);
    if (!env || !env[0]) return;

    const std::string value(env);
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(':', start);
        const size_t stop = end == std::string::npos ? value.size() : end;
        if (stop > start) {
            paths.emplace_back(value.substr(start, stop - start));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}
