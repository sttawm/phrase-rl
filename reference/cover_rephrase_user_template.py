    def get_rephrase_batch(self, instruction, batch_number=1):
        instruction = f"""
            Given the original instruction: "{instruction}", and the appeneded image, generate {batch_number} reworded instructions that convey the same objective.

            Guidelines for rephrasing:
            1. Use simple, clear words and actions (focus on verbs and nouns)
            2. Remove adverbs whenever possible
            3. Keep descriptions concise but complete
            4. Infer and include object colors when they can be reasonably deduced (e.g., apples are typically red, strawberries are red)
            5. Use diverse vocabulary across rephrases (vary nouns, verbs, and adjectives)
            6. Ensure each rephrase maintains the same core meaning and task objective
            7. Try to generate as diverse as possible rephrases.
            8. Consider the image when generating the rephrased instructions.
            
            Examples:
            Original: "put apple on the desk"
            Reworded: "pick up the red apple and place it on the desk", "take the apple and put it on the desk", "place the red fruit on the desk"
            
            Original: "put cooking pot in the green basket"
            Reworded: "move the silver cooking pot to the green basket", "take the cooking pot and put it in the green basket", "put the utensil into the green basket"
            
            Original: "put strawberry on top of the fridge"
            Reworded: "put the red fruit on the fridge", "place the red berry on the top of the fridge", "set the red berry on the top of the refrigerator"
            
            Original: "lift the water bottle and place it on the desk"
            Reworded: "pick up the transparent bottle and place it on the wooden desk", "take the hydration bottle and put it on the desk", "place the water on the desk"
            
            
            Guidelines for generation: 
            1. You need to consider both image and instruction when generating the rephrased instructions.
            2. You need to first generate a description of the image in your own words, and then think about what does the language instruction mean in the context of the image.
            
            
            Format your response as:
            <Description of the image>
            <Meaning of the instruction in the context of the image>
            Original: <Nouns> as many as possible potential replacements: <Nouns>
            Original: <Verbs> as many as possible potential replacements: <Verbs>
            Original: <Adjectives> as many as possible potential replacements: <Adjectives>
            Original: <Adverbs>

            Original Instruction:
            {instruction}

            Reworded Instructions:
            1. <Alternative phrasing 1>
            2. <Alternative phrasing 2>
            ...
            {batch_number}. <Alternative phrasing {batch_number}>
            
            Important: Ensure all rephrased instructions avoid adverbs, use diverse vocabulary, and maintain the same objective as the original.
            """
        return instruction
