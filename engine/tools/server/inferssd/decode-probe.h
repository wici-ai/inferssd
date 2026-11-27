// SPDX-License-Identifier: Apache-2.0
#pragma once

struct common_params;
struct server_context;

// --ssd-decode-probe: loads the model and context of PARAMS as the server loads them to serve, but with every routed
// expert run by the CPU expert sink, measures what a decode step costs apart from its routed experts and prints it as
// one JSON line on stdout. Returns the exit status: 0, or 1 after saying why on stderr.
int server_decode_probe(server_context & ctx_server, common_params & params);
