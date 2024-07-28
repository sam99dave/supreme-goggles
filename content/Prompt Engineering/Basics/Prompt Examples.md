
> There are examples for multiple tasks over [here](https://www.promptingguide.ai/introduction/examples). I'm adding only those which I found unique & interesting!

### Role Prompting

Perhaps one of the more interesting things you can achieve with prompt engineering is instructing the LLM system on how to behave, its intent, and its identity. This is particularly useful when you are building conversational systems like customer service chatbots.

**Prompt**

```
The following is a conversation with an AI research assistant. The assistant tone is technical and scientific.

Human: Hello, who are you?
AI: Greeting! I am an AI research assistant. How can I help you today?
Human: Can you tell me about the creation of blackholes?
AI:
```

**Output**

```
Sure! Black holes are regions of spacetime where the gravitational force is so strong that nothing, not even light, can escape from it. They are created when a very massive star dies and its core collapses in on itself, forming a singularity of infinite density. The intense gravity of the singularity pulls in all the matter and radiation around it, creating the black hole.
```


### Code Generation

**Prompt**

```
/*
Ask the user for their name and say "Hello"
*/
```

**Output** (ChatGPT)

```Python
# Ask the user for their name
name = input("What is your name? ")

# Greet the user with "Hello"
print(f"Hello, {name}!")

```


