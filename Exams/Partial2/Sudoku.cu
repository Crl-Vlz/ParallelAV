
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>

#define uchar unsigned char

// Sudoku input
// Changed from int to uchar for less storage space
const uchar board[9][9] = { {5, 3, 0, 0, 7, 0, 0, 0, 0},
                 {6, 0, 0, 1, 9, 5, 0, 0, 0},
                 {0, 9, 8, 0, 0, 0, 0, 6, 0},
                 {8, 0, 0, 0, 6, 0, 0, 0, 3},
                 {4, 0, 0, 8, 0, 3, 0, 0, 1},
                 {7, 0, 0, 0, 2, 0, 0, 0, 6},
                 {0, 6, 0, 0, 0, 0, 2, 8, 0},
                 {0, 0, 0, 4, 1, 9, 0, 0, 5},
                 {0, 0, 0, 0, 8, 0, 0, 7, 9} };

/// <summary>
/// Checks general validity of board using host functions
/// </summary>
/// <param name="cols">gets number of columns</param>
/// <param name="rows">gets number of rows</param>
/// <returns>Valid or not</returns>
bool host_check_general(int cols, int rows) {
    if (cols != 9) {
        printf("\nBoard does not respect 9x9 format");
        return 0;
    }
    if (rows != 9) {
        printf("\nBoard does not respect 9x9 format");
        return 0;
    }
    return 1;
}

/// <summary>
/// Host checks validity of the n row
/// </summary>
/// <param name="board">1D array that represents the board</param>
/// <param name="row">nth row</param>
/// <returns>Valid or not</returns>
bool host_check_row(uchar * board, int row) {
    int starter = row * 9 ; // Passes n values to start a new row
    bool row_values[9] = { false };
    for (int i = 0; i < 9; i++) {
        if (board[i + starter] == 0) continue;
        if (row_values[board[i + starter] - 1]) return 0;
        row_values[board[i + starter] - 1] = true;
    }
    return true;
}

/// <summary>
/// Host checks validity of the nth column
/// </summary>
/// <param name="board">1D array that represents the sudoku board</param>
/// <param name="col">nth column of the board</param>
/// <returns>Valid or not</returns>
bool host_check_col(uchar * board, int col) {
    int starter = col;
    bool col_values[9] = { false };
    for (int i = 0; i < 9; i++) {
        if (board[i * 9 + starter] == 0) continue;
        if (col_values[board[i * 9 + starter] - 1]) return 0;
        col_values[board[i * 9 + starter - 1]];
    }
    return true;
}

/// <summary>
/// Host checks validity of the nth subsquare [0-8]
/// </summary>
/// <param name="board">1D array that represents the sudoku board</param>
/// <param name="sub">nth subspace of the board</param>
/// <returns>Was the subspace valid or not</returns>
bool host_check_subsquare(uchar* board, int sub) {
    int col = (sub % 3) * 3, row = (sub / 3) * 3;
    bool square_values[9] = { false };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int starter = (row + i) * 9 + col + j;
            if (board[starter] == 0) continue;
            if (square_values[board[starter] - 1]) return 0;
            square_values[board[starter] - 1] = true;
        }
    }
    return 1;
}

/// <summary>
/// Host checks validity of the subsquare that hosts the cell (col, row)
/// </summary>
/// <param name="board">1D array that represents the sudoku board</param>
/// <param name="col">nth column</param>
/// <param name="row">nth row</param>
/// <returns>Was the subsace valid or not</returns>
bool host_check_subsquare(uchar* board, int col, int row) {
    int sub = col / 3 + (row / 3) * 3;
    return host_check_subsquare(board, sub);
}

/// <summary>
/// Realizes all the checks of the board and returns if the board was valid or not
/// </summary>
/// <param name="board">1D array that represents the sudoku board</param>
/// <returns>Is the board valid or not</returns>
bool host_check_all(uchar* board) {
    bool format = true;
    for (int i = 0; i < 9; i++) {
        format = format && host_check_row(board, i);
    }
    for (int i = 0; i < 9; i++) {
        format = format && host_check_col(board, i);
    }
    for (int i = 0; i < 9; i++) {
        format = format && host_check_subsquare(board, i);
    }
    return format;
}

/// <summary>
/// Check if the number n is valid in the cell (col, row)
/// </summary>
/// <param name="board">1D array that represents the sudoku board</param>
/// <param name="n">A number between 1 and 9</param>
/// <param name="col">Column to place <i>n<i></param>
/// <param name="row">Row to place <i>n<i></param>
/// <returns></returns>
bool host_check_validity(uchar* board, int n, int col, int row) {
    for (int i = 0; i < 9; i++)
        if (board[row * 9 + i] == n || board[i * 9 + col] == n) return false;
    int sub_row_start = (row / 3) * 3;
    int sub_col_start = (col / 3) * 3;
    for (int i = sub_row_start; i < sub_row_start + 3; i++)
        for (int j = sub_col_start; j < sub_col_start + 3; j++)
            if (board[i * 9 + j] == n) return false;
    return true;
}

/// <summary>
/// A recursive backtracking solution implemented in host to solce a sudoku board
/// </summary>
/// <param name="board">1D array that represents the sudoku board</param>
/// <returns>-1 invalid format. 0 no solution. 1 solution was found</returns>
int host_solve_sudoku(uchar* board) {
    if (!host_check_all(board)) return -1; // Board is not valid
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            int index = col + row * 9;
            if (!board[index]) {
                for (int i = 1; i <= 9; i++) {
                    if (host_check_validity(board, i, col, row)) {
                        board[index] = i;
                        if (host_solve_sudoku(board)) return 1;
                        else board[index] = 0;
                    }
                }
                return 0;
            }
        }
    }
    return 1;
}

/// <summary>
/// Helper function to print a pointer with newlines
/// </summary>
/// <param name="pointer">Pointer to print</param>
/// <param name="size">Number of elements of the pointer</param>
/// <param name="max_inline">Number of elements in each line before a newline</param>
void print_pointer(uchar* pointer, int size, int max_inline = -1) {
    if (max_inline <= 0) {
        for (int i = 0; i < size; i++) printf("\n%d", pointer[i]);
    }
    else {
        int flag = size / max_inline;
        int vals = size % max_inline == 0 ? flag : flag + 1;
        for (int i = 0; i < vals; i++) {
            printf("\n");
            for (int j = 0; j < max_inline; j++) {
                if (j + i * max_inline >= size) return;
                printf("%d ", pointer[j + i * max_inline]);
            }
        }
    }
}

/// <summary>
/// A GPU helper function to check validity of the nth column
/// </summary>
/// <param name="board">1D array representing the sudoku board</param>
/// <param name="col">The nth column of the board</param>
/// <returns>Was the column valid or not</returns>
__device__ bool device_check_col(uchar* board, int col) {
    bool col_values[9] = {false};
    for (int i = 0; i < 9; i++) {
        if (!board[i * 9 + col]) continue;
        if (col_values[board[i * 9 + col] - 1]) return false;
        else col_values[board[i * 9 + col] - 1] = true;
    }
    return true;
}

/// <summary>
/// A GPU helper function to check validity of the nth row
/// </summary>
/// <param name="board">1D array representing the sudoku board</param>
/// <param name="row">The nth row of the board</param>
/// <returns>Was the row valid or not</returns>
__device__ bool device_check_row(uchar* board, int row) {
    bool row_values[9] = { false };
    for (int i = 0; i < 9; i++) {
        if (!board[row * 9 + i]) continue;
        if (row_values[board[row * 9 + i] - 1]) return false;
        else row_values[board[row * 9 + i] - 1] = true;
    }
    return true;
}

/// <summary>
/// A GPU helper function to check if the subspace is valid or not
/// </summary>
/// <param name="board">1D array representing the sudoku board</param>
/// <param name="square">nth subsquare of the board</param>
/// <returns>Was the subsquare valid or not</returns>
__device__ bool device_check_subsquare(uchar* board, int square) {
    bool square_values[9] = { false };
    int row = (square / 3) * 3, col = (square % 3) * 3;
    int start_cell = row * 9 + col;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int index = start_cell + i * 9 + j;
            if (!board[index]) continue;
            if (square_values[board[index] - 1]) return false;
            else square_values[board[index] - 1] = true;
        }
    }
    return true;
}

/// <summary>
/// Kernel to check all rows, columns and subsquares.
/// Uses logical AND operations to prevent overwriting.
/// With a native true value, if it gets false it will always remain false
/// </summary>
/// <param name="board">1D array representing the sudoku board</param>
/// <param name="status">Was the board valid or not</param>
/// <returns>Void</returns>
__global__ void device_validity_all(uchar* board, bool* status) {
    int gid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int oid = gid; //Operation id
    if (gid >= 27) return;
    if (gid < 9) {
        status&& device_check_col(board, oid);
    }
    else if (gid < 18) {
        //TODO device check row
        oid -= 9;
        status&& device_check_row(board, oid);
    }
    else if (gid < 27) {
        //TODO device check subsquare
        oid -= 18;
        status&& device_check_subsquare(board, oid);
    }
}

/// <summary>
/// GPU helper function to check if the value n is valid in the cell (row, col)
/// </summary>
/// <param name="board">1D array representing the sudoku board</param>
/// <param name="row">the nth row of the board</param>
/// <param name="col">the nth column of the board</param>
/// <param name="n">An int value from 1 to 9</param>
/// <returns>If the number is valid at that cell</returns>
__device__ bool device_is_valid(uchar* board, int row, int col, int n) {
    // Check all cells in a row
    for (int i = 0; i < 9; i++) if (board[row * 9 + i] == n) return false;
    // Check all cells in a column
    for (int i = 0; i < 9; i++) if (board[i * 9 + col] == n) return false;
    // Check all cells in a subsquare
    int startRow = row / 3 * 3;
    int startCol = col / 3 * 3;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if (board[(i + startRow) * 9 + (j + startCol)] == n) return false;
    return true;
}

/// <summary>
/// A kernel that tries to solve the sudoku board via recursive backtracking
/// Check if this method works in GPU, if not look for another
/// </summary>
/// <param name="board">1D array representing the sudoku board</param>
/// <param name="found">Was a solution found</param>
/// <returns>Void</returns>
__global__ void device_solve_sudoku(uchar* board, bool* found) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (*found) return;
    if (gid >= 81) return;

    int row = gid / 9, col = gid % 9;

    if (board[gid] == 0) {
        for (int num = 1; num <= 9; num++) {
            board[gid] = num;
            __syncthreads();
            if (device_is_valid(board, row, col, num)) {
                board[gid] = num;
                device_solve_sudoku << <1, 1 >> > (board, found);
                if (*found == 1) return;
                board[gid] = 0;
            }
            else {
                board[gid] = 0;
            }
            __syncthreads();
        }
    }
    else {
        device_solve_sudoku << <1, 1 >> > (board, found);
        __syncthreads();
        if (gid == 80) return;
    }

    if (row == 8 && col == 8 && board[row * 9 + col] != 0) *found = 1;
}

int main()
{
    uchar* host_pointer;
    uchar* device_pointer;
    uchar* sudoku_board;
    // 81 because sudoku boards always have 81 values
    host_pointer = (uchar*)malloc(sizeof(uchar) * 81);
    cudaMalloc((void**)&device_pointer, sizeof(uchar) * 81);
    sudoku_board = (uchar*)malloc(sizeof(uchar) * 81);
    for (int i = 0; i < 81; i++) {
        int row = i / 9;
        int col = i % 9;
        host_pointer[i] = board[row][col];
        sudoku_board[i] = board[row][col];
    }
    
    clock_t start, end;
    double time_used;

    cudaMemcpy(device_pointer, host_pointer, sizeof(uchar) * 81, cudaMemcpyHostToDevice);
    start = clock();
    int result = host_solve_sudoku(host_pointer);
    if (result == 1) {
        printf("\nValid format");
        printf("\nSudoku board solved\n");
        print_pointer(host_pointer, 81, 9);
    }
    else if (result == 0) {
        printf("\nValid format");
        printf("\nThere was no solution");
    }
    else {
        printf("\nFormat was invalid");
    }
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("\nTotal Time CPU: %lf", time_used);

    bool status = true;
    // Size of warp
    device_validity_all << <1, 32 >> > (device_pointer, &status);
    cudaDeviceSynchronize();
    if (!status) 
    {
        printf("\nThe format of the board is not valid");
        return 0;
    }
    else printf("\nFormat is valid");
    bool solved = false;
    // Three warps
    device_solve_sudoku << <1, 96 >> > (device_pointer, &solved);
    cudaMemcpy(host_pointer, device_pointer, sizeof(uchar) * 81, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if (!solved)
    {
        printf("\nNo solution was found");
        return 0;
    }
    else printf("\nSolution was found");
    print_pointer(host_pointer, 81, 9);
    free(host_pointer);
    free(sudoku_board);
    cudaFree(device_pointer);
}
