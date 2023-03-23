/*
 Copyright 2023 Adrien Roussel <adrien.roussel@protonmail.com>
 SPDX-License-Identifier: CECILL-C
*/

#ifndef COMMON_H
#define COMMON_H

#include <time.h>

#define handle_error_en(en, msg) \
  do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)

#ifdef __cplusplus
extern "C"
{
#endif
  /* return time in seconds */
  double get_elapsedtime(void);
#ifdef __cplusplus
}
#endif

#endif
