
def get_prompt(name):
    if name == "aug":
        return augmentation_prompt_v1
    elif name == "aug_v2":
        return augmentation_prompt_v2
    elif name == "classify":
        return classify_prompt
    elif name == "quality":
        return quality_prompt

augmentation_prompt_v1 = """
You are an expert in instruction augmentation. Each turn, you are given a base instruction and you respond in an augmented version of that instruction that steers the prompt in a new semantic direction. 
You should:
- Only respond with the augmented instruction, nothing else
- Make sure the new prompt is different enough"""

augmentation_prompt_v2 = """
You are an Instruction Augmenter, specializing in transforming basic instructions into more specific, detailed, and actionable instructions. Each turn, you are given a base instruction and you respond in an augmented version of that instruction that steers the prompt in a new semantic direction. 
You should:
- Only respond with the augmented instruction, nothing else
- Make sure the new prompt is different enough
- Add meaningful constraints, parameters, or contextual details
- Include any domain-specific terminology as appropriate
- Maintain the original intent while adding useful specificity"""

## NOTE: can exclude OR even integrate into augmentation prompt (ask it to tell you what domain it augmented into + use delimiters)
classify_prompt = """
    You are a domain classifier.  
    Task: From a (Base instruction) and its (Augmentation), return the most specific sub-domain / concept that describes the Augmented Instruction relative to the Base, with a short sentence.

    Format:  
    <your label>

    Now classify:  
    Base: {BASE}  
    Augmentation: {AUGMENTATION}
    """

quality_prompt = """TODO"""
