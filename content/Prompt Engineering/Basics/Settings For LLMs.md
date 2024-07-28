
### Temperature

In terms of application, you might want to use a lower temperature value for tasks like fact-based QA to encourage more factual and concise responses. For poem generation or other creative tasks, it might be beneficial to increase the temperature value.

### Top P (Nucleus Sampling)

- Instead of selecting the next word based on the highest probability alone, nucleus sampling considers a subset of the vocabulary.
- This subset is chosen so that the cumulative probability of the words in the subset is greater than or equal to a threshold p and the subset is the smallest.
- Once this subset (or "nucleus") is formed, the next word is sampled from this subset, introducing a level of randomness.

Vocab set ~ 50k, then the probability of giving the next word for a randomly initialized model is 1/50k. When `p` is set low then the threshold is met early and with more factual and deterministic responses.

> The general recommendation is to alter temperature or Top P but not both

### Frequency Penalty

Applies a penalty on the next token proportional to how many times that token already appeared in the response and prompt.

> Reduces the repetition of words in the model's response by giving tokens that appear more a higher penalty.

### Presence Penalty

Applies a penalty on repeated tokens but, unlike the frequency penalty, the penalty is the same for all repeated tokens. A token that appears twice and a token that appears 10 times are penalized the same. This setting prevents the model from repeating phrases too often in its response.

`Creative Text` - High Penalty

`Stay Focused` - Low Penalty

> Similar to `temperature` and `top_p`, the general recommendation is to alter the frequency or presence penalty but not both.

### Other Settings

- Max Length
- Stop Sequences



`TOC` - [[Intro]]


