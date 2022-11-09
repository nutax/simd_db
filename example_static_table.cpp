// -------------------------------------
// AUTHORS
// -------------------------------------
/* Created by José Ignacio Huby Ochoa */

// -------------------------------------
// DEPENDENCIES
// -------------------------------------
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "simd_db.hpp"

// -------------------------------------
// DATA
// -------------------------------------

/* CONSTANTS */
#define CACHE_LINE 64
#define VEC_SIZE 32
#define MAX_PLAYERS 30
#define MAX_DOORS 100

/* STATIC ALLOCATIONS */
// Players
enum
{
    PLAYER_X,
    PLAYER_Y,
    PLAYER_Z,
    PLAYER_TEAM,
    PLAYER_OTHER_DATA
};
simd_db::static_table<CACHE_LINE, VEC_SIZE, MAX_PLAYERS, float, float, float, uint32_t, uint32_t> players;

// Doors
enum
{
    DOOR_X,
    DOOR_Y,
    DOOR_Z,
    DOOR_R2,
    DOOR_TEAM,
    DOOR_OPEN
};
simd_db::static_table<CACHE_LINE, VEC_SIZE, MAX_DOORS, float, float, float, float, uint32_t, uint32_t> doors;

// -------------------------------------
// PROCEDURES
// -------------------------------------

/* INLINE */
#define CHECK_TIME(_name, ...)                                                                              \
    {                                                                                                       \
        struct timespec start, end;                                                                         \
        clock_gettime(CLOCK_REALTIME, &start);                                                              \
        __VA_ARGS__;                                                                                        \
        clock_gettime(CLOCK_REALTIME, &end);                                                                \
        double f = ((double)end.tv_sec * 1e9 + end.tv_nsec) - ((double)start.tv_sec * 1e9 + start.tv_nsec); \
        printf(_name " time %f ms\n", f / 1000000);                                                         \
    }

/* DECLARATIONS */
int main(int argc, char **argv);
void generate();
void open_doors();

/* DEFINITIONS */
int main(int argc, char **argv)
{
    CHECK_TIME("GENERATE", generate());
    CHECK_TIME("OPEN DOORS", open_doors());
    return EXIT_SUCCESS;
}

void generate()
{
    srand(time(NULL));

    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        players.create<PLAYER_X>() = (float)(rand() % 5);
        players.create<PLAYER_Y>() = (float)(rand() % 5);
        players.create<PLAYER_Z>() = (float)(rand() % 5);
        players.create<PLAYER_TEAM>() = (uint32_t)(rand() % 3);
        players.create();
    }
    for (int i = 0; i < MAX_DOORS; ++i)
    {
        doors.create<DOOR_X>() = (float)(rand() % 5);
        doors.create<DOOR_Y>() = (float)(rand() % 5);
        doors.create<DOOR_Z>() = (float)(rand() % 5);
        doors.create<DOOR_R2>() = (float)(rand() % 25);
        doors.create<DOOR_TEAM>() = (uint32_t)(rand() % 3);
        doors.create<DOOR_OPEN>() = 0;
        doors.create();
    }
}

void open_doors()
{
    for (int i = 0; i < doors.size(); i += doors.v_step<DOOR_X>())
    {
        __m256 vdx = _mm256_load_ps(doors.column<DOOR_X>() + i);
        __m256 vdy = _mm256_load_ps(doors.column<DOOR_Y>() + i);
        __m256 vdz = _mm256_load_ps(doors.column<DOOR_Z>() + i);
        __m256 vdr = _mm256_load_ps(doors.column<DOOR_R2>() + i);
        __m256i vdt = _mm256_load_si256((__m256i *)(doors.column<DOOR_TEAM>() + i));
        __m256i vdo = _mm256_setzero_si256();

        for (int j = 0; j < players.size(); ++j)
        {
            __m256 vpx = _mm256_broadcast_ss(players.column<PLAYER_X>() + j);
            __m256 vpy = _mm256_broadcast_ss(players.column<PLAYER_Y>() + j);
            __m256 vpz = _mm256_broadcast_ss(players.column<PLAYER_Z>() + j);
            __m256i vpt = _mm256_set1_epi32(players.column<PLAYER_TEAM>()[j]);

            __m256 xdiff = _mm256_sub_ps(vdx, vpx);
            __m256 ydiff = _mm256_sub_ps(vdy, vpy);
            __m256 zdiff = _mm256_sub_ps(vdz, vpz);

            __m256 xdist2 = _mm256_mul_ps(xdiff, xdiff);
            __m256 ydist2 = _mm256_mul_ps(ydiff, ydiff);
            __m256 zdist2 = _mm256_mul_ps(zdiff, zdiff);

            __m256 dist2 = _mm256_add_ps(_mm256_add_ps(xdist2, ydist2), zdist2);

            __m256 dist_mask = _mm256_cmp_ps(dist2, vdr, _CMP_LE_OS);
            __m256i team_mask = _mm256_cmpeq_epi32(vdt, vpt);
            __m256i result_mask = _mm256_and_si256(_mm256_castps_si256(dist_mask), team_mask);

            vdo = _mm256_or_si256(vdo, result_mask);
        }
        _mm256_store_si256((__m256i *)(doors.column<DOOR_OPEN>() + i), vdo);
    }
}