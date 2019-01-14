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

## Sequence Definition

The mode is the delimiter between the **Parameters** and the **Sequence Definition**. If it is not specified, it is assumed to be Sequence 1 (`:`), and all code is assumed to be part of the sequence definition. The **Sequence Definition** is a comma (`,`) delimited list of **terms**.

### Terms

The interpreter loops through each term specified, restarting at the beginning of the list when it reaches the end, until the output (based on the mode) has been found. Terms evaluate to numbers, and follow usual order of operations (currently). Terms can contain the following (will be expanded on), and are evaluated seperately:

#### Numbers and Variables

The interpreter reads a number when it reaches a `.` or a digit.

The following variables are currently valid identifiers:

| Name | Syntax | Description
| ---- | ------ | -----------
| Input | `A` to `E` | Gets the first (`A`) to fifth (`E`) input (not including `n`; see **Parameters.Input** section for more details)
| Previous | `v` to `z` | Gets the most recent (`z`) to fifth most recent (`v`) terms in the sequence. This means terms that have already been calculated. Defaults to `0`.
| Current | `$` | Gets what term we are currently calculating in the sequence, starting with the starting index, if set, or `1`.

#### Constants

`_` signifies that the next char will be a constant identifier. The following are currently valid constants:

| Name | Syntax | ~Value | Description
| ---- | ------ | ------ | -----------
| pi | `_p` | `3.141592653589793` | Currently equals Python's `math.pi`
| e | `_e` | `2.718281828459045` | Currently equals Python's `math.e`

#### Operations

| Type | Name | Syntax | Description
| ---- | ---- | ------ | -----------
| Container | Parentheses | `(`code`)`\* | Acts like usual mathematical parentheses: surrounds a group of code and elevates it in the order of operations
| Binary | Exponentation | `^` | Returns the left side to the power of the right side
| Binary | Scientific | `e` | Returns the left side times ten to the power of the right side
| Binary | Addition | `+` | Adds the left side to the right side
| Binary | Subtraction | `-` | Subtracts the right side from the left side
| Binary | Multiplication | `*`, implicit | Multiplies the left side by the right side; can be used implicitly, as in Algebra: `2z`, `2(1+$)`
| Binary | Division | `/` | Divides the left side by the right side
| Binary | Integer Division | `//` | Integer divides the left side by the right side
| Binary | Modulus | `%` | Gets the left side mod the right side
| Unary | Negation | `-` | Negates the right side (only occurs when there is no left side)

\* note that trailing `)`s right before the end of your code can all be left off

#### Builtins

Builtin function are called with their identifier, a comma-delimited list of their parameters, and a closing parenthesis (`)`). Remember that trailing `)`s can be left off.

| Name | Identifier | Parameters | Description
| ---- | ---------- | ---------- | -----------
| Absolute Value | `a` | 1 | Returns the absolute value of the first parameter
| Ceiling | `c` | 1 | Returns the ceil of the first parameter
| Factorial | `f` | 1 | Returns the factorial (`!`) of the first parameter
| Floor | `F` | 1 | Returns the floor of the first parameter
| OEIS | `O<index><letter>` | 1? | Returns the item in the index of the first parameter in the OEIS sequence `<letter><possible leading zeroes><index>`, if implemented (see `oeis.py`)
| Logarithm | `l` | 1 or 2 | If there is no second parameter, returns the natural logarithm (base `e` logarithm) of the first parameter. Otherwise, returns the base [second-parameter] logarithm of the first parameter
| Multiline | `\0` to `\9` | 1 to many | Returns the result of the cQuents program at the (0-based) index of the function, with the function parameters passed as input to it
| Next Prime | `p` | 1 | Returns the next prime **after** the first parameter
| Root | `r` | 1 or 2 | If there is no second parameter, returns the square root of the first parameter. Otherwise, returns the first parameter to the power of the reciprocal of the second parameter
| Cosine | `\c` | 1 | Returns the cos of the first parameter
| Logarithm 2 | `\l` | 1 | Returns the base `10` logarithm of the first parameter
| Sine | `\s` | 1 | Returns the sin of the first parameter
| Tangent | `\t` | 1 | Returns the tan of the first parameter

**Multiline Builtins**

`\0` to `\9` call the cQuents program on the (0-indexed) line with the input given and return its output.

**OEIS**

Currently, there are plans for cQuents to have a dictionary of some OEIS sequences (stored as cQuents generator programs) which can be called as builtin functions. For those sequences currently implemented (see `oeis.py` for a full list), the sequence can be called with the function `O#A`, where `#` is the sequence index with all leading zeroes removed, and `A` is the alphabetic letter preceding the sequence (as of 7/24/2017 all OEIS sequences use `A`, but once the numeric index recents, the assumption is that it will start at `B`). Pass this function the (usually 1-based) index you want in the sequence - see its line in `oeis.py` for details on exactly how it is implemented in cQuents.

## Parameters

Before the mode is specified, you are not building your sequence. Instead, you are setting it up. You can setup your sequence and how it will be used in several ways.

### Literals

You can specify three types of literals: Front, between, and back. These literals will not be calculated in your sequence, but will be printed with it. Until any other parameter is specified, all characters except those with significance before the mode is set are by default literals. To force a character to be a literal, you can prepend it with an `@` symbol to escape it, or surround a string with `'` to escape the whole string.

When specifying literals, they are in the format `<front>|<between>|<back>`. You can stop early, and skip them. To set just the front literal to `1`, just `1` will suffice. To set just the back literal to `1`, you can use `||1`. To set all three to `1`, use `1|1|1`.

Literals are used when displaying your output. Front literals are prepended to your output, and back literals are appended to it. The default between literal is `,`. The between literal replaces `,` as the list delimiter.

Once `#`, `=`, `$`, or `"` are read, the literals are done.

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
