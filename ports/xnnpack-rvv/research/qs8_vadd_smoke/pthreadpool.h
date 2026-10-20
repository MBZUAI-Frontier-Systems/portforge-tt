// xnnpack.h includes <pthreadpool.h> for one opaque handle type; microkernels
// never call into it, so the smoke builds against this stub instead.
#pragma once
typedef struct pthreadpool* pthreadpool_t;
