
### @register_passable("trivial")

- Its [[Trivial]]
- Its [[Register Passable]]

When a type is marked as `@register_passable("trivial")`, it tells Mojo that the type should be passed in machine registers whenever possible, which has clear performance benefits. It also indicates that the type should be copyable and movable but that it has no user-defined logic (no [[Lifecycle Methods]]) for doing this.

```python
@register_passable("trivial")  
struct Pair:  
    var a: Int  
    var b: Int  
```

In this example, `Pair` is a trivial type that can be passed in machine registers. It's important to note that when using `@register_passable("trivial")`, the only [[Lifecycle Methods]] you're allowed to define is the `__init__()` constructor. You _cannot_ define any copy or move constructors or a destructor.


### @always_inline

The `@always_inline` decorator in Modular's Mojo language is used to instruct the Mojo compiler to "inline" the body of a function directly into the body of the calling function. 

>This means the function's code is copied and placed directly into the calling function, eliminating the need for a function call.

This can potentially improve performance by avoiding the overhead of function calls, which involve jumping to a new point in the code. Normally, the compiler will automatically inline functions where it can improve performance, but the `@always_inline` decorator forces it to do so.

```python
@always_inline  
fn add(a: Int, b: Int) -> Int:  
    return a + b  
  
print(add(1, 2))  
```

The above is equivalent to:

```python
print(1 + 2)  
```

**Limitation**
- However, there's a trade-off: in-lining can increase the binary size by duplicating the function at every call site.

You can also use `@always_inline` with the `"nodebug"` argument, which in-lines the function but without debug information. This means you can't step into the function when debugging. 
This is useful for low-level functions in a library, which may wrap primitive functions, MLIR operations, or inline assembly.