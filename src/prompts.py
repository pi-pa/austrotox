

class Prompts:
            
    system_msg_toxicity_old = """Du bekommst den Titel eines Zeitungsartikels und einen Benutzerkommentar zum Artikel.
    Ist der Kommentar toxisch? 
    Definition: Ein beleidigender oder toxischer Kommentar beinhaltet verunglimpfende Äußerungen gegenüber Personen, Personengruppen oder anderen Entitäten. Er kann auch zu Hass oder Gewalt gegen einzelne Personen oder Personengruppen aufrufen.
    Antworte ausschliesslich im JSON-Format mit der folgenden Struktur: {"Label": <0 or 1>}"""
    
    system_msg_toxicity_german = """Du bekommst den Titel eines Zeitungsartikels und einen Benutzerkommentar zum Artikel.
    Ist der Kommentar toxisch? 
    Definition: Ein beleidigender oder toxischer Kommentar beinhaltet verunglimpfende Äußerungen gegenüber Personen, Personengruppen oder anderen Entitäten. Er kann auch zu Hass oder Gewalt gegen einzelne Personen oder Personengruppen aufrufen.
    Antworte entweder mit "1" (int), für toxisch, oder mit "0" (int), für nicht-toxisch.
    Andere Antworten sind nicht erlaubt.
    """
    
    system_msg_toxicity_english = """You get the title of a newspaper article and a user comment on the article.
    Is the comment offensive?
    Definition: An offensive comment contains derogatory remarks about persons, groups of persons or other entities. It can also call for hatred or violence against individuals or groups of persons.
    Respond either with "1" (int), for offensive, or with "0" (int), for non-offensive.
    No other answers are allowed.
    """

    system_msg_multitask_german = """Du bekommst den Titel eines Zeitungsartikels und einen Benutzerkommentar zum Artikel. Deine Aufgabe ist es, folgende Fragen zum Kommentar zu beantworten:
    1. Ist der Kommentar toxisch? (1 = toxisch, 0 = nicht-toxisch) 
    Definition: Ein beleidigender oder toxischer Kommentar beinhaltet verunglimpfende Äußerungen gegenüber Personen, Personengruppen oder anderen Entitäten. Er kann auch zu Hass oder Gewalt gegen einzelne Personen oder Personengruppen aufrufen.
    2. Wer oder was ist das Ziel der Toxizität? 
    Markiere mindestens eines der folgenden Ziele der Toxizität, falls der Kommentar toxisch ist: "Target_Group", "Target_Individual", oder "Target_Other".
    3. Markiere Vulgaritäten mit "Vulgarity". Vulgaritäten können in toxischen und nicht-toxischen Kommentaren vorkommen.
    Antworte im JSON-Format mit den folgenden Feldern:
    ```json
    {
        "Label": <0 or 1>,
        "Tags": [
            {
                "Tag": <"Target_Group", "Target_Individual", "Target_Other", or "Vulgarity">,
                "Token": <Span des Targets oder der Vulgarity>
            },
            ...
        ]
    }
    ```
    """

    system_msg_toxicity_german_vulgar = """Du bekommst den Titel eines Zeitungsartikels und einen Benutzerkommentar zum Artikel. Deine Aufgabe ist es, folgende Frage zum Kommentar zu beantworten:
    Ist der Kommentar toxisch? (1 = toxisch, 0 = nicht-toxisch) 
    Definition: Ein beleidigender oder toxischer Kommentar beinhaltet verunglimpfende Äußerungen gegenüber Personen, Personengruppen oder anderen Entitäten. Er kann auch zu Hass oder Gewalt gegen einzelne Personen oder Personengruppen aufrufen.
    Vulgaritäten können in toxischen und nicht-toxischen Kommentaren vorkommen. Sie sind mit <vulgarity>vulgärer Ausdruck</vulgarity> markiert.
    Antworte mit 0 oder 1.
    """


    system_msg_multitask_english = """You receive a user comment from a news paper article. Your task is to answer the following questions about the comment:
    1. Is the comment toxic? (1 = toxic, 0 = non-toxic)
    Definition: An offensive or toxic comment includes derogatory remarks towards individuals, groups of people, or other entities. It may also incite hatred or violence against individual persons or groups of people.
    2. Who or what is the target of toxicity?
    Mark at least one of the following targets of toxicity if the comment is toxic: "Target_Group", "Target_Individual", or "Target_Other".
    3. Mark vulgarities with "Vulgarity". Vulgarities can occur in toxic and non-toxic comments.
    Respond in JSON format with the following fields:
    ```json
    {
        "Label": <0 or 1>,
        "Tags": [
            {
                "Tag": <"Target_Group", "Target_Individual", "Target_Other", or "Vulgarity">,
                "Token": <Span of the target or the vulgarity>
            },
            ...
        ]
    }
    ```
    """
    
    examples_german = """
    Hier sind fünf Beispiele mit Antworten:
    Titel 1: {title_1}
    Kommentar 1: {comment_1}
    Antwort 1: {response_1}
    Titel 2: {title_2}
    Kommentar 2: {comment_2}
    Antwort 2: {response_2}
    Titel 3: {title_3}
    Kommentar 3: {comment_3}
    Antwort 3: {response_3}
    Titel 4: {title_4}
    Kommentar 4: {comment_4}
    Antwort 4: {response_4}
    Titel 5: {title_5}
    Kommentar 5: {comment_5}
    Antwort 5: {response_5}
    """

    examples_english = """
    Here are five examples with responses:
    Comment 1: {comment_1}
    Response 1: {response_1}
    Comment 2: {comment_2}
    Response 2: {response_2}
    Comment 3: {comment_3}
    Response 3: {response_3}
    Comment 4: {comment_4}
    Response 4: {response_4}
    Comment 5: {comment_5}
    Response 5: {response_5}
    """

    user_msg_template_5shot_german = """
    Jetzt kommt das zu klassifizierende Beispiel:
    Titel 6: {title}
    Kommentar 6: {comment}
    Antwort 6: """

    user_msg_template_5shot_english = """
    Now here comes the example to be classified:
    Comment 6: {comment}
    Response 6: """

    user_msg_template_0shot_german = """Titel: {title}
    Kommentar: {comment}
    Antwort: """

    user_msg_template_0shot_english = """Comment: {comment}
    Response: """


class ChatGPTPrompts(Prompts):
    
    @classmethod
    def get_toxicity_system_msg(cls, num_shots, shots, language):
        if num_shots == 0:
            if language == "de":
                return cls.system_msg_toxicity_german
            elif language == "en":
                return cls.system_msg_toxicity_english
            else:
                raise ValueError(f"Unexpected language: {language}")
        elif num_shots == 5:
            if language == "de":
                return cls.system_msg_toxicity_german + cls.examples_german.format(**shots)
            elif language == "en":
                return cls.system_msg_toxicity_english + cls.examples_english.format(**shots)
            else:
                raise ValueError(f"Unexpected language: {language}")
        else:
            raise ValueError(f"Unexpected number of shots: {num_shots}")
    
    @classmethod
    def get_multitask_system_msg(cls, num_shots, shots, language):
        if num_shots == 0:
            if language == "de":
                return cls.system_msg_multitask_german
            elif language == "en":
                return cls.system_msg_multitask_english
            else:
                raise ValueError(f"Unexpected language: {language}")
        elif num_shots == 5:
            if language == "de":
                return cls.system_msg_multitask_german + cls.examples_german.format(**shots)
            elif language == "en":
                return cls.system_msg_multitask_english + cls.examples_english.format(**shots)
            else:
                raise ValueError(f"Unexpected language: {language}")
        else:
            raise ValueError(f"Unexpected number of shots: {num_shots}")
    
    @classmethod
    def get_user_msg(cls, title, comment, num_shots, language):
        if num_shots == 0:
            if language == "de":
                return cls.user_msg_template_0shot_german.format(title=title, comment=comment)
            elif language == "en":
                return cls.user_msg_template_0shot_english.format(comment=comment)
            else:
                raise ValueError(f"Unexpected language: {language}")
        elif num_shots == 5:
            if language == "de":
                return cls.user_msg_template_5shot_german.format(title=title, comment=comment)
            elif language == "en":
                return cls.user_msg_template_5shot_english.format(comment=comment)
            else:
                raise ValueError(f"Unexpected language: {language}")
        else:
            raise ValueError(f"Unexpected number of shots: {num_shots}")


class LEOLMPrompts(Prompts):
    
    @classmethod
    def get_toxicity_system_msg(cls, num_shots, shots, language):
        if num_shots == 0:
            if language == "de":
                return "\n<|im_start|>system\n" +  cls.system_msg_toxicity_german  + "<|im_end|>"
            elif language == "en":
                return "\n<|im_start|>system\n" +  cls.system_msg_toxicity_english + "<|im_end|>"
            else:
                raise ValueError(f"Unexpected language: {language}")
        elif num_shots == 5:
            if language == "de":
                return "\n<|im_start|>system\n" + cls.system_msg_toxicity_german + cls.examples_german.format(**shots) + "<|im_end|>"
            elif language == "en":
                return "\n<|im_start|>system\n" + cls.system_msg_toxicity_english + cls.examples_english.format(**shots) + "<|im_end|>"
            else:
                raise ValueError(f"Unexpected language: {language}")
        else:
            raise ValueError(f"Unexpected number of shots: {num_shots}")
    
    @classmethod
    def get_multitask_system_msg(cls, num_shots, shots, language):
        if num_shots == 0:
            if language == "de":
                return "\n<|im_start|>system\n" +  cls.system_msg_multitask_german + "<|im_end|>"
            elif language == "en":
                return "\n<|im_start|>system\n" +  cls.system_msg_multitask_english + "<|im_end|>"
            else:
                raise ValueError(f"Unexpected language: {language}")
        elif num_shots == 5:
            if language == "de":
                return "\n<|im_start|>system\n" + cls.system_msg_multitask_german + cls.examples_german.format(**shots) + "<|im_end|>"
            elif language == "en":
                return "\n<|im_start|>system\n" + cls.system_msg_multitask_english + cls.examples_english.format(**shots) + "<|im_end|>"
            else:
                raise ValueError(f"Unexpected language: {language}")
        else:
            raise ValueError(f"Unexpected number of shots: {num_shots}")
    
    @classmethod
    def get_user_msg(cls, title, comment, num_shots, language):
        if num_shots == 0:
            if language == "de":
                return "\n<|im_start|>user\n" +  cls.user_msg_template_0shot_german.format(title=title, comment=comment) + "<|im_end|>" + "\n<|im_start|>assistant\n"
            elif language == "en":
                return "\n<|im_start|>user\n" +  cls.user_msg_template_0shot_english.format(comment=comment) + "<|im_end|>"  + "\n<|im_start|>assistant\n"
            else:
                raise ValueError(f"Unexpected language: {language}")
        elif num_shots == 5:
            if language == "de":
                return "\n<|im_start|>user\n" +  cls.user_msg_template_5shot_german.format(title=title, comment=comment) + "<|im_end|>" + "\n<|im_start|>assistant\n"
            elif language == "en":
                return "\n<|im_start|>user\n" +  cls.user_msg_template_5shot_english.format(comment=comment) + "<|im_end|>" + "\n<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unexpected number of shots: {num_shots}")
    
    @staticmethod
    def combine_system_and_user_msg(system_msg, user_msg):
        return system_msg + user_msg
