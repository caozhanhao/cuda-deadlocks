#pragma once

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

// Small, host-only parser. Bind arguments to variables that outlive parse_args.
// --names are optional named arguments; other names are optional positionals.
// A bool binding is a flag. String/integer bindings consume one value.
class ArgumentParser {
    struct Argument {
        std::string name, help, default_value;
        bool flag, positional;
        std::function<bool(std::string_view)> assign;
        std::function<std::string()> value;
    };
    std::vector<Argument> arguments_;
    std::string program_ = "program";

    void add(std::string name, std::string help, bool flag,
             std::function<bool(std::string_view)> assign, std::function<std::string()> value) {
        if (name.empty() || name == "--help" || name == "-h") error("Invalid argument name");
        for (const auto& argument : arguments_) {
            if (argument.name == name) error("Duplicate argument: " + name);
        }
        const bool positional = name.rfind("--", 0) != 0;
        if (positional && name.front() == '-') error("Use a --name for options: " + name);
        if (flag && positional) error("Boolean flags need a --name: " + name);
        const auto default_value = value();
        arguments_.push_back({std::move(name), std::move(help), default_value, flag, positional,
                              std::move(assign), std::move(value)});
    }

  public:
    // Initial variable values are the defaults. Optional bounds are checked
    // while parsing, before setup can initialize CUDA.
    template <class T>
    void add_argument(std::string name, T& target, std::string help,
                      std::common_type_t<T> minimum = std::numeric_limits<T>::lowest(),
                      std::common_type_t<T> maximum = std::numeric_limits<T>::max()) {
        static_assert(std::is_integral_v<T>, "Use a bool, integer, or std::string binding");
        add(
            std::move(name), std::move(help), std::is_same_v<T, bool>,
            [&target, minimum, maximum](std::string_view text) {
                if constexpr (std::is_same_v<T, bool>) {
                    target = true;
                } else {
                    T parsed{};
                    const auto result =
                        std::from_chars(text.data(), text.data() + text.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
                        parsed < minimum || parsed > maximum)
                        return false;
                    target = parsed;
                }
                return true;
            },
            [&target]() -> std::string {
                if constexpr (std::is_same_v<T, bool>)
                    return target ? "true" : "false";
                else
                    return std::to_string(target);
            });
    }

    void add_argument(std::string name, std::string& target, std::string help,
                      std::initializer_list<const char*> choices = {}) {
        std::vector<std::string> allowed(choices.begin(), choices.end());
        if (!allowed.empty()) {
            help += " (choices:";
            for (const auto& choice : allowed) help += " " + choice;
            help += ")";
        }
        add(
            std::move(name), std::move(help), false,
            [&target, allowed](std::string_view text) {
                const std::string value(text);
                if (!allowed.empty() &&
                    std::find(allowed.begin(), allowed.end(), value) == allowed.end())
                    return false;
                target = value;
                return true;
            },
            [&target] { return target; });
    }

    void print_help(FILE* output = stdout) const {
        std::fprintf(output, "Usage: %s [-h|--help]", program_.c_str());
        for (const auto& argument : arguments_) {
            std::fprintf(output, " [%s%s]", argument.name.c_str(),
                         argument.flag || argument.positional ? "" : " VALUE");
        }
        std::fprintf(output, "\n\n  -h, --help               Show this help and exit\n");
        for (const auto& argument : arguments_) {
            std::fprintf(output, "  %-24s %s (default: %s)\n", argument.name.c_str(),
                         argument.help.c_str(), argument.default_value.c_str());
        }
    }

    [[noreturn]] void error(const std::string& message) const {
        std::fprintf(stderr, "%s: %s\n", program_.c_str(), message.c_str());
        print_help(stderr);
        std::exit(EXIT_FAILURE);
    }

    void parse_args(int argc, char** argv) {
        program_ = argc > 0 ? argv[0] : "program";
        size_t positional = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string token(argv[i]);
            if (token == "--help" || token == "-h") {
                print_help();
                std::exit(EXIT_SUCCESS);
            }
            Argument* argument = nullptr;
            std::string value;
            if (token.rfind("--", 0) == 0) {
                const auto separator = token.find('=');
                const auto name = token.substr(0, separator);
                for (auto& candidate : arguments_) {
                    if (!candidate.positional && candidate.name == name) argument = &candidate;
                }
                if (!argument) error("Unknown option: " + name);
                if (argument->flag) {
                    if (separator != std::string::npos) error("Flag takes no value: " + name);
                } else if (separator != std::string::npos) {
                    value = token.substr(separator + 1);
                } else {
                    if (i + 1 == argc) error("Missing value for " + name);
                    const std::string_view next(argv[i + 1]);
                    if (next.rfind("--", 0) == 0 || next == "-h")
                        error("Missing value for " + name);
                    value = argv[++i];
                }
            } else {
                while (positional < arguments_.size() && !arguments_[positional].positional)
                    ++positional;
                if (positional == arguments_.size()) error("Unexpected argument: " + token);
                argument = &arguments_[positional++];
                value = token;
            }
            if (!argument->assign(value))
                error("Invalid value for " + argument->name + ": " + value);
        }
    }

    void print_values() const {
        for (const auto& argument : arguments_)
            std::printf("ARG %s=%s\n", argument.name.c_str(), argument.value().c_str());
    }
};
