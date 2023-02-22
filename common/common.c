/*
 Copyright 2023 Adrien Roussel <adrien.roussel@protonmail.com>
 SPDX-License-Identifier: CECILL-C
*/

#ifdef __cplusplus
extern "C"
{
#endif

  #include "common.h"

  /* return time in second */
  double get_elapsedtime(void)
  {
    struct timespec st;
    int err = gettime(&st);
    if (err !=0) return 0;
    return (double)st.tv_sec + get_sub_seconde(st);
  }
#ifdef __cplusplus
}
#endif
