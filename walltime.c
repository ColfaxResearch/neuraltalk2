//compiled with C_INCLUDE_PATH=~/torch/install/include gcc -Wall -shared -fPIC -o walltime.so -llua-5.1 walltime.c
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include "omp.h"
static int walltime(lua_State * L) {
    double time = omp_get_wtime();
    lua_pushnumber(L, time);
    return 1;
}

int luaopen_walltime(lua_State * L) {
    lua_register(L, "walltime", walltime);
    return 0;
}
