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
#define MAX_PLAYERS 30
#define MAX_DOORS 100

/* COLUMNS */
COLUMN(xpos, float);
COLUMN(ypos, float);
COLUMN(zpos, float);
COLUMN(r2, float);
COLUMN(team, uint32_t);
COLUMN(open, uint32_t);
COLUMN(other, uint32_t);

/* TABLES */
// Players
TABLE256(players, MAX_PLAYERS, col_xpos, col_ypos, col_zpos, col_team, col_other);

// Doors
TABLE256(doors, MAX_DOORS, col_xpos, col_ypos, col_zpos, col_r2, col_team, col_open);

/* VIEWS */
VIEW(with_position, std::ref(table_players), std::ref(table_doors));

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
        table_players.create<col_xpos>() = (float)(rand() % 5);
        table_players.create<col_ypos>() = (float)(rand() % 5);
        table_players.create<col_zpos>() = (float)(rand() % 5);
        table_players.create<col_team>() = (uint32_t)(rand() % 3);
        table_players.create();
    }
    for (int i = 0; i < MAX_DOORS; ++i)
    {
        table_doors.create<col_xpos>() = (float)(rand() % 5);
        table_doors.create<col_ypos>() = (float)(rand() % 5);
        table_doors.create<col_zpos>() = (float)(rand() % 5);
        table_doors.create<col_r2>() = (float)(rand() % 25);
        table_doors.create<col_team>() = (uint32_t)(rand() % 3);
        table_doors.create<col_open>() = 0;
        table_doors.create();
    }
}

void open_doors()
{
    for (int i = 0; i < table_doors.size(); i += table_doors.v_step<col_xpos>())
    {
        __m256 vdx = _mm256_load_ps(table_doors.column<col_xpos>() + i);
        __m256 vdy = _mm256_load_ps(table_doors.column<col_ypos>() + i);
        __m256 vdz = _mm256_load_ps(table_doors.column<col_zpos>() + i);
        __m256 vdr = _mm256_load_ps(table_doors.column<col_r2>() + i);
        __m256i vdt = _mm256_load_si256((__m256i *)(table_doors.column<col_team>() + i));
        __m256i vdo = _mm256_setzero_si256();

        for (int j = 0; j < table_players.size(); ++j)
        {
            __m256 vpx = _mm256_broadcast_ss(table_players.column<col_xpos>() + j);
            __m256 vpy = _mm256_broadcast_ss(table_players.column<col_ypos>() + j);
            __m256 vpz = _mm256_broadcast_ss(table_players.column<col_zpos>() + j);
            __m256i vpt = _mm256_set1_epi32(table_players.column<col_team>()[j]);

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
        _mm256_store_si256((__m256i *)(table_doors.column<col_open>() + i), vdo);
    }
}