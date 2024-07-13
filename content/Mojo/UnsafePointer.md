
[Documentation](https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/)

```python
from memory.unsafe_pointer import UnsafePointer, initialize_pointee_copy, initialize_pointee_move 
# Allocate memory to hold a value 
var ptr = UnsafePointer[Int].alloc(1) 
# Initialize the allocated memory 
initialize_pointee_copy(ptr, 100) 
# Update an initialized 
value ptr[] += 10 # Access an initialized value print(ptr[])
```

Unsafe pointers in Modular are a type that holds an address to memory. You can store and retrieve values in that memory.
These are generic and can point to any type of value, the type of the value is specified in the parameter.

Notes:
- Memory management is up to the user. You need to manually allocate and free memory, and be aware of when other APIs are allocating or freeing memory for you.
- `UnsafePointer` and `DTypePointer` values are _nullable_—that is, the pointer is not guaranteed to point to anything. And even when a pointer points to allocated memory, that memory may not be _initialized_.
- Mojo doesn't track lifetimes for the data pointed to by an `UnsafePointer`. When you use an `UnsafePointer`, managing memory and knowing when to destroy objects is your responsibility.

```python
from memory.unsafe_pointer import UnsafePointer, initialize_pointee_copy, initialize_pointee_move

# Allocate memory to hold a value
var ptr = UnsafePointer[Int].alloc(1)

# # Initialize the allocated memory
# initialize_pointee_copy(ptr, 100)
print(ptr[])

# Update an initialized value
ptr[] += 10

# Access an initialized value
print(ptr[])
```

Over here the memory pointed has not been initialised and allocated. The output for this case is as follows:

```bash
101923235127384 
101923235127394
```


**unsafe** - This implements `various types` that can work with unsafe pointers
**unsafe_pointer** - This implements a generic `unsafe pointer type` 



