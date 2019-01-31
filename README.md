# cQuents
An esoteric programming language designed for golfing. It can describe mathematical sequences and series with very little source code.

My interpreter's code was originally modeled after [this very helpful tutorial](https://ruslanspivak.com/lsbasi-part1/), but it has been heavily modified since then.

To try cQuents, [click here for an online interpreter](https://tio.run/#cquents). Note that the online interpreter may be behind the master branch.

# Table of Contents

- [Basic Tutorial](https://github.com/stestoltz/cQuents#basic-tutorial)
  - [Input](https://github.com/stestoltz/cQuents#input)
  - [Modes](https://github.com/stestoltz/cQuents#modes)
  - [Sequence Definition](https://github.com/stestoltz/cQuents#sequence-definition)
    - [Terms](https://github.com/stestoltz/cQuents#terms)
      - [Numbers and Variables](https://github.com/stestoltz/cQuents#numbers-and-variables)
      - [Constants](https://github.com/stestoltz/cQuents#constants)
      - [Operations](https://github.com/stestoltz/cQuents#operations)
      - [Builtins](https://github.com/stestoltz/cQuents#builtins)
      - [Conditionals](https://github.com/stestoltz/cQuents#conditionals)
      - [Lists](https://github.com/stestoltz/cQuents#lists)
    - [Parameters](https://github.com/stestoltz/cQuents#parameters)
      - [Literals](https://github.com/stestoltz/cQuents#literals)
      - [Default Input](https://github.com/stestoltz/cQuents#default-input)
      - [Sequence Start](https://github.com/stestoltz/cQuents#sequence-start)
      - [Starting Index](https://github.com/stestoltz/cQuents#starting-index)
      - [Stringify](https://github.com/stestoltz/cQuents#stringify)
   - [Multiline](https://github.com/stestoltz/cQuents#mulitline)
   - [Examples](https://github.com/stestoltz/cQuents#examples)

# Basic Tutorial

##### Note that this tutorial may be out of date - see commands.txt for a more up-to-date list of commands

## Input

Input is currently given as a space delimited list of numbers. Your program will can query the input (for more explanation, see **Sequence Definition.Terms.Numbers and Variables**, as well as **Parameters.Default Input**). The interpreter will check how deep into the input you query. The last input is optional, and occurs after all other input is allocated to the interpreter from your queries. `n` is used in different ways by different modes, and is `1`-indexed. `n`s existance determines your program's functionality, based on the mode.

## Modes

cQuents is designed with sequences in mind. Before you start creating a cQuents program, you need to select a mode. Currently, there are four modes:

| Name | Syntax | Description
| ---- | ------ | -----------
| Sequence 1 | `:` | If there is `n`, output the `n`th term; if no `n`, output the whole sequence
| Sequence 2 | `::` | If there is `n`, output the sequence up to `n`; if no `n`, TODO: converge or diverge
| Series | `;` | If there is `n`, output the sum of the sequence (or "series") up to `n`; if no `n`, output the whole series
| Query | `?` | If there is `n`, output `true` if `n` is in the sequence, and `false` if `n` is not in the sequence; if no `n`, ignore right of mode
| Reverse Query | `??` | If there is `n`, output `false` if `n` is in the sequence, and `true` if `n` is not in the sequence; if no `n`, ignore right of mode

## Sequence Definition

The mode is the delimiter between the **Parameters** and the **Sequence Definition**. If it is not specified, it is assumed to be Sequence 1 (`:`), and all code is assumed to be part of the sequence definition. The **Sequence Definition** is a comma (`,`) delimited list of **terms**.

### Terms

The interpreter loops through each term specified, restarting at the beginning of the list when it reaches the end, until the output (based on the mode) has been found. Terms evaluate to numbers, and follow usual order of operations (currently). Terms can contain the following (will be expanded on), and are evaluated seperately:

#### Numbers and Variables

The interpreter reads a number when it reaches a `.` or a digit.

The following variables are currently valid identifiers:

| Name | Syntax | Description
| ---- | ------ | -----------
| Input | `A` to `C` | Gets the first (`A`) to third (`C`) input (not including `n`; see **Parameters.Input** section for more details)
| Previous | `X` to `Z` | Gets the most recent (`Z`) to third most recent (`X`) terms in the sequence. This means terms that have already been calculated. Defaults to `0`.
| Current | `$` | Gets what term we are currently calculating in the sequence, starting with the starting index, if set, or `1`.
| n | `n` | Gets n, if provided
| k | `k` | Starts at 1 and increments whenever the terms list resets
| Ten | `t` | Evaluates to 10
| Smallest | `w` | Evaluates to the smallest positive integer not yet in the sequence

#### Constants

The character `` ` `` signifies that the next char will be a constant identifier. The following are currently valid constants:

| Name | Syntax | ~Value | Description
| ---- | ------ | ------ | -----------
| 1/2 | `` `2`` | `1/2` | 1/2
| 1/3 | `` `3`` | `1/3` | 1/3
| 1/4 | `` `4`` | `1/4` | 1/4
| 1/10 | `` `0`` | `1/10` | 1/10
| Catalan's constant | `` `c`` | `0.915965594177219` | Sum of 1 - 1/9 + 1/25 - 1/49 + ...
| e | `` `e`` | `2.718281828459045` | Currently equals Python's `math.e`
| Golden Ratio | `` `g`` | `1.618033988749895` | `.5 * (math.sqrt(5) + 1)`
| Glaisher-Kinkelin constant | `` `G`` | `0.128242712910062` | .
| Khinchin's constant | `` `k`` | `0.268545200106530` | .
| pi | `` `p`` | `3.141592653589793` | Currently equals Python's `math.pi`
| Euler-Mascheroni constant | `` `y`` | `0.577215664901532` | .

#### Operations

For simplicity, in the descriptions, the expression to the left of the operator is called `x` and the expression to the left of the operator is called `y`.

| Type | Name | Syntax | Description
| ---- | ---- | ------ | -----------
| Container | Parentheses | `(`code`)`\* | Acts like usual mathematical parentheses: surrounds a group of code and elevates it in the order of operations
| Unary | Negation | `-` | Negates `y` (only occurs when there is no `x`)
| Unary | Positive | `+` | Positive of `y` (only occurs when there is no `x`, can be used to convert non-numbers to numbers)
| Unary | Bitwise Not | `~` | The negative of (`y` plus one) (only occurs when there is no `x`)
| Unary | Reciprocal | `/` | One divided by `y` (only occurs when there is no `x`)
| Unary | Logical Not | `!` | Logical negation of `x`
| Post-Unary | Factorial | `!` | The factorial of `x`
| Binary | Addition | `+` | Adds `x` to `y`
| Binary | Subtraction | `-` | Subtracts `y` from `x`
| Binary | Concatenation | `~` | Concatenates `y` to `x`
| Binary | Multiplication | `*`, implicit | Multiplies `x` by `y`; can be used implicitly, as in Algebra: `2z`, `2(1+$)`
| Binary | Division | `/` | Divides `x` by `y`
| Binary | Exponentiation | `^` | Returns `x` to the power of `y`
| Binary | Modulus | `%` | Gets `x` mod `y`
| Binary | Scientific | `e` | Returns `x` times ten to the power of `y`
| Binary | Equality | `=` | Python: `x == y`
| Binary | Less than | `<` | Python: `x < y`
| Binary | Greater than | `>` | Python: `x > y`
| Binary | And | `&` | Python: `x and y`
| Binary | Or | `\|` | Python: `x or y`
| Binary | In | `i` | Python: `x in y`
| Ternary | If-Else | `?:` | Like Java ternary - `<condition>?<value if true>:<value if false>`

\* note that trailing `)`s right before the end of your code can all be left off

##### Extra Operations

The character `_` signifies an two-byte operation.

| Type | Name | Syntax | Description
| ---- | ---- | ------ | -----------
| Unary | Rotate Left | `_l` | Rotates `y` left once
| Unary | Rotate Right | `_r` | Rotates `y` right once
| Unary | Increment | `_+` | Adds one to `y`
| Unary | Decrement | `_-` | Subtracts one from `y`
| Binary | Integer Division | `_/` | Integer divides `x` by `y`
| Binary | Bitwise OR | `_\|` | Python: `x | y`
| Binary | Bitwise NOR | `_n` | Python: `~(x | y)`
| Binary | Bitwise XOR | `_^` | Python: `x ^ y`
| Binary | Bitwise XNOR | `_x` | Python: `~(x ^ y)`
| Binary | Bitwise AND | `_&` | Python: `x & y`
| Binary | Bitwise NAND | `_N` | Python: `~(x & y)`
| Binary | Bitwise Left Shift | `_{` | Python: `x << y`
| Binary | Bitwise Right Shift | `_}` | Python: `x >> y`
| Binary | Inequality | `_=` | Python: `x != y`
| Binary | Less than or equal to | `_<` | Python: `x < y`
| Binary | Greater than or equal to | `_>` | Python: `x > y`

##### Operator Precedence Table

| First |
| ----- |
| `?:` |
| Post-Unary Ops, `[]` |
| Unary Ops |
| `^`, `e` |
| `*`, `/`, `_/`, `%` |
| `-`, `+`, `~` |
| `_{`, `_}` |
| `_&`, `_N` |
| `_^`, `_x` |
| `_\|`, `_n` |
| `=`, `_=`, `<`, `_<`, `>`, `_>`, `i` |
| `&` |
| `\|` |
| **Last** |

#### Builtins

Builtin function are called with their identifier, a `;`-delimited list of their parameters, and a closing parenthesis (`)`). Remember that trailing `)`s can be left off.

For simplicity, in the descriptions, the individual parameters given to the functions are `a,b,c, ...`.

| Name | Identifier | Parameters | Description
| ---- | ---------- | ---------- | -----------
| First Line | `a` | 1 to many | Returns the first line given the parameters as input
| Second Line | `b` | 1 to many | Returns the second line given the parameters as input
| Third Line | `c` | 1 to many | Returns the third line given the parameters as input
| Line Function | `d` | 2 to many | Returns the line at `a` given the rest of the parameters
| Digits | `D` | 1 | Returns the digit list of `a`
| Factorial | `f` | 1 | Returns the factorial (`!`) of `a`
| Floor | `F` | 1 | Returns the floor of `a`
| Char | `h` | 1 | Returns Python's `chr` of `a`
| Input Function | `I` | 1 | Returns the input at the position of `a`
| Join | `j` | 1 or 2 | Returns `a` joined on `b` (if there is no `b`, defaults to `""`)
| Logarithm | `l` | 1 or 2 | If `b` exists, returns the natural logarithm (base `e` logarithm) of `a`. Otherwise, returns the base `b` logarithm of `a`
| Length | `L` | 1 or 2 | Returns the length of `a`, including the decimal point if `b` exists
| Min | `m` | 1 | Returns the minimum of `a`
| Max | `M` | 1 | Returns the maximum of `b`
| OEIS | `O<index><letter>` | 1? | Returns the item in the index of `a` in the OEIS sequence `<letter><possible leading zeroes><index>`, if implemented (see `oeis.py`)
| Ordinal | `o` | 1 | Returns Python's `ord` of `a`
| Next Prime | `p` | 1 | Returns the next prime **after** `a`
| Previous Function | `P` | 1 | Returns the previous term back as many terms as `a`
| Prime Factors | `q` | 1 | Returns the prime factors of `a`
| Deduplicate | `Q` | 1 | Removes duplicates from `a`, retaining order
| Root | `r` | 1 or 2 | If `b` does not exist, returns the square root of `a`. Otherwise, returns `a` to the power of the reciprocal of `b`
| Round | `R` | 1 | Returns `a`, rounded
| Sort | `s` | 1 | Returns `a`, sorted
| String | `S` | 1 | Returns `a` converted to a string
| Ceiling | `T` | 1 | Returns the ceil of `a`
| Count | `u` | 2 | Returns the count of `b` in `a`
| Sum | `U` | 1 | Returns the sum of `a`
| Absolute Value | `v` | 1 | Returns the absolute value of `a`
| Average | `V` | 1 | Returns the average value of `a`
| Exp | `x` | 1 | Returns e to the power of `a`
| Divisors | `z` | 1 | Returns the divisors of `a`
| Cosine | `\c` | 1 | Returns the cos of `a`
| Logarithm 2 | `\l` | 1 | Returns the base `10` logarithm of `a`
| Reverse | `\r` | 1 or 2 | Returns the reverse of `a`, including the decimal point if `b` exists
| Rotate | `\R` | 2 or 3 | Returns `a` rotated by `b`, including the decimal point if `c` exists
| Sine | `\s` | 1 | Returns the sin of `a`
| Tangent | `\t` | 1 | Returns the tan of `a`
| Smallest | `\w` | 1 | Returns the `a`th smallest positive integer not yet in the sequence
| Proper Divisors | `\z` | 1 | Returns the proper divisors of `a`

Note: you can use `E` in front of a builtin to apply the builtin to every parameter. For example, `EL1;11;111)` will give `1,2,3`.

**OEIS**

Currently, there are plans for cQuents to have a dictionary of some OEIS sequences (stored as cQuents generator programs) which can be called as builtin functions. For those sequences currently implemented (see `oeis.py` for a full list), the sequence can be called with the function `O#A`, where `#` is the sequence index with all leading zeroes removed, and `A` is the alphabetic letter preceding the sequence (as of 7/24/2017 all OEIS sequences use `A`, but once the numeric index resets, the assumption is that it will start at `B`). Pass this function the (usually 1-based) index you want in the sequence - see its line in `oeis.py` for details on exactly how it is implemented in cQuents.

#### Conditionals

Conditionals are started with `#` and ended with `)`. Conditionals use the variable `N`, which is the number it is checking. They return `N` when all `;` separated terms in the conditional are true.

#### Lists

Lists are denoted by `{}` and separated by `;`. Operators performed on lists that error will try again, this time applied to all of the list's members. Lists can be indexed and sliced like lists in Python.

## Parameters

Before the mode is specified, you are not building your sequence. Instead, you are setting it up. You can setup your sequence and how it will be used in several ways.

### Literals

You can specify three types of literals: Front, between, and back. These literals will not be calculated in your sequence, but will be printed with it. Until any other parameter is specified, all characters except those with significance before the mode is set are by default literals. To force a character to be a literal, you can prepend it with an `@` symbol to escape it, or surround a string with `'` to escape the whole string.

When specifying literals, they are in the format `<front>|<between>|<back>`. You can stop early, and skip them. To set just the front literal to `1`, just `1` will suffice. To set just the back literal to `1`, you can use `||1`. To set all three to `1`, use `1|1|1`.

Literals are used when displaying your output. Front literals are prepended to your output, and back literals are appended to it. The default between literal is `,`. The between literal replaces `,` as the list delimiter.

Once `#`, `=`, `$`, or `"` are read, the literals are done.

If literals are used (with either `@` or `'`) after the above symbols are read, they will act like strings, and the interpreter will attempt to work with them as strings.

### Default Input

`#` signifies that the list following it will be default input. Default input is combined with user input to form the total input, which must align with the expected input (which is based on the highest input requested by the **Sequence Definition**). This is the best way for you to create a constant.

The default input can be split in two with `|`. The list on the left side of `|` is prepended to the user-provided input, while the list on the right side is appended to the user-provided input. Remember, the if there is an extra input, the last input in the combined list becomes `n`.

### Sequence Start

`=` signifies that the list following it will be the beginning terms of the sequence. These will be the terms returned if `n` falls into their range. The previous variables, `z` to `q`, may evaluate to parts of the sequence start early on in the sequence.

### Starting Index

`$` allows you to change the starting index in the sequence. The default is `1`, with an increment of `1`. The first parameter passed to `$` will be the new starting index, and the optional second parameter passed to `$` will be the new increment. Make sure that this increment will reach `n` if you are setting `n` - there is currently no checking except for equality, though this may change in the future.

### Stringify

`"` signifies that if the final sequence will be treated list a string instead of a list. Instead of getting the `n`th term in the list, the interpreter concatenate the list together and return the `n`th char in the string. Alternatively, instead of printing the list delimited by `,`s, it prints the list as a string without delimiter, unless the between literal is specified.

## Multiline

You can include multiple cQuents programs in one file, on different lines. The program on the first line is the main program, which is called when the interpreter is executed. It outputs to STDOUT. You can reference other programs by using the builtin functions `\0` to `\9`, which call the cQuents program on the (0-indexed) line, passing the input to that cQuents program. Functions `\0` to `\9` output by returning to the main program - they do **not** output to STDOUT.

As a side note, be wary of your multiline indexing when using literal newlines (before the mode) as they may mess up your internal indexing. A literal newline will not affect multiline parsing or indexing.

# Examples

Using default mode (`:`) for all examples, which will return the sequence if given no `n`, or the `n`th item if given `n`. Mode can modified without affecting the sequence. Explanation follows sequence.

Positive Integers: `1, 2, 3, 4, 5, ...`

```
$   

Implicit mode :. Term equals what number term we are in.
```

Even Numbers: `2, 4, 6, 8, 10, ...`

```
2$  

Implicit mode :. Term equals 2 times what number term we are in.
```

Fibonacci Sequence: `0, 1, 1, 2, 3, 5, ...`

```
=0,1:z+y

Seed sequence with 0,1. Term equals last term plus second-to-last term.
```

Squares: `1, 4, 9, 16, 25, ...`

```
$$

Implicit mode :. Term equals current term times current term.
```

One Third: `.3333333...`

```
.":3

Literal . before sequence. Stringify sequence. Each term equals 3.
```

Primes: `2, 3, 5, 7, 11, ...`

```
pz

Implicit mode :. p builtin with parameter z get the next prime after the previous term. Implicit ).
```
