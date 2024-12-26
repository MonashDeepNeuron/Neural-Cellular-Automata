/**
 * @param {character[][]} board
 * @return {boolean}
 */

/**
Edges cases: 
Store an array for:
1. Columns
2. Rows
3. Grids (hash the consective number to the
 
Each hashmap has nine entries, each is a set (O(9) max)

Assuming row major order for the grids
Each hashmap holds 1...9 and 
Iterate from i = 0...8 
    J = 0...8
        i = row
        j = col
        k = i//3 * 3 + j//3 
 */

/**
 * Learning lesson:
 * const columns = Array(9).fill(new Set()),
    rows = Array(9).fill(new Set()),
    grids = Array(9).fill(new Set()); 

 Does not create 27 separate Set instances, rather creates a new Set() then passes that reference
 to each element in the array

 Use map() or loop to ensure new Set item is created 
 */
var isValidSudoku = function (board) {

    const seen = new Set();

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {

            const x = board[i][j];
            if (x != ".") {

                const k = Math.floor(i / 3) * 3 + Math.floor(j / 3);
                const row_hash = i.toString() + "R" + x,
                    col_hash = j.toString() + "C" + x,
                    grid_hash = k.toString() + "G" + x;

                if (seen.has(row_hash)) {
                    return false;
                } else {
                    seen.add(row_hash);
                }

                if (seen.has(col_hash)) {
                    return false;
                } else {
                    seen.add(col_hash);
                }

                if (seen.has(grid_hash)) {
                    return false;
                } else {
                    seen.add(grid_hash);
                }

            }
        }

    }

    return true;

};

const board =
    [[".", ".", "4", ".", ".", ".", "6", "3", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", "."],
    ["5", ".", ".", ".", ".", ".", ".", "9", "."],
    [".", ".", ".", "5", "6", ".", ".", ".", "."],
    ["4", ".", "3", ".", ".", ".", ".", ".", "1"],
    [".", ".", ".", "7", ".", ".", ".", ".", "."],
    [".", ".", ".", "5", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", ".", "."]];

console.log(isValidSudoku(board));

/** 
 * Interesting solution: using strings
 * 
 * Use a "hashing scheme" prefix all row groups by a i( all col groups by j[ all grid groups by k{
 *  */