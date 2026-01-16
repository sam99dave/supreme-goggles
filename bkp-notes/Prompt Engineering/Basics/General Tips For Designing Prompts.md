
`Note` - It is really an iterative process that requires a lot of experimentation to get optimal results.

> These are tips & recommendations, its important to experiment with these.

**For Big / Complex Tasks**

Break down the task into simpler sub tasks and keep building up as you get better results. This avoids adding too much complexity to the prompt design process at the beginning.

### Instruction

- You can design effective prompts for various simple tasks by using commands to instruct the model what you want to achieve
- Recommend that you place instructions at the beginning of the prompt.
- Another recommendation is to use some clear separator like "###" to separate the instruction and context.


### Specificity

- Be very specific about the instruction and task you want the model to perform.
- The more descriptive and detailed the prompt is, the better the results.
- Providing examples in the prompt is very effective to get desired output in specific formats.

> If you want the LLM to give the proper format, e.g. classify as neutral and not Neutral. Few shot prompting is the recommended approach!


### Something Interesting

Here is an example of a movie recommendation chatbot failing at exactly what I don't want it to do because of how I wrote the instruction -- focusing on what not to do.

**Prompt**

```bash
The following is an agent that recommends movies to a customer. DO NOT ASK FOR INTERESTS. DO NOT ASK FOR PERSONAL INFORMATION.

Customer: Please recommend a movie based on my interests.
Agent: 
```

**Output**

```bash
Sure, I can recommend a movie based on your interests. What kind of movie would you like to watch? Do you prefer action, comedy, romance, or something else?
```

***Better Prompt***

```
The following is an agent that recommends movies to a customer. The agent is responsible to recommend a movie from the top global trending movies. It should refrain from asking users for their preferences and avoid asking for personal information. If the agent doesn't have a movie to recommend, it should respond "Sorry, couldn't find a movie to recommend today.".

Customer: Please recommend a movie based on my interests.
Agent:
```

**Output**

```
Sorry, I don't have any information about your interests. However, here's a list of the top global trending movies right now: [list of movies]. I hope you find something you like!
```

