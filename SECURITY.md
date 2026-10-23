# Security policy

The inference server binds to loopback by default and is intended to run on a trusted machine. Do not expose
it directly to an untrusted network. Model files and configuration files must come from trusted sources.

Use this repository's private vulnerability-reporting channel when it is available. Do not put credentials,
model data, deployment addresses or exploit details in a public issue. While private reporting is unavailable,
contact the repository administrators through an existing private channel.

No released InferSSD version has been qualified yet. Security fixes will target the current development branch
until the first release establishes the supported-version policy.
