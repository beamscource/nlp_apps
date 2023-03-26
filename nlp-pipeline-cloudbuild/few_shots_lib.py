'''A collection of few-shot examples which can be used with the Aleph Alpha completion API.
As the default, we try to define prompts which include at least 10 examples.'''

def summary_shots():

    few_shots = """
    ###
    Text: We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state 3D model Our network learns to predict the object's pose part segmentation and corresponding motion parameters to reproduce the articulation shown in the input image
    Summary: We present a neural network which learns o predict the object's pose part segmentation and corresponding motion parameters
    ###
    Text: Unlike language tasks where the output space is usually limited to a set of tokens the output space of visual tasks is more complicated making it difficult to build a unified visual model for various visual tasks In this paper we seek to unify the output space of visual tasks so that we can also build a unified model for visual tasks
    Summary: We seek to unify the output space of visual tasks where the output space is more complicated unlike in language tasks where the output space is limited
    ###
    Text: Autoformalization seeks to address this by translating proofs written in natural language into a formal representation that is computer-verifiable via interactive theorem provers In this paper we introduce a semantic parsing approach based on the Universal Transformer architecture that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover
    Summary: We introduce a semantic parsing approach based on the Universal Transformer architecture to translate proofs written in natural language into a formal representation
    ###
    Text: In this paper we explored the use of deep learning for the prediction of aortic flow metrics obtained using 4D flow MRI using wearable seismocardiography devices We hypothesized that deep learning could be used to identify pathological changes in blood flow
    Summary: We explored the use of deep learning for the prediction of aortic flow metrics obtained using 4D flow MRI using wearable seismocardiography devices
    ###
    Text: Fashion-image editing represents a challenging computer vision task where the goal is to incorporate selected apparel into a given input image Most existing techniques known as Virtual Try-On methods deal with this task by first selecting an example image of the desired apparel and then transferring the clothing onto the target person Conversely in this paper we consider editing fashion images with text descriptions
    Summary: In this paper we consider editing fashion images with text descriptions to solve the challenging computer vision task of fashion-image editing
    ###
    Text: Developing agents that can execute multiple skills by learning from pre-collected datasets is an important problem in robotics where online interaction with the environment is extremely time-consuming In this work we propose a novel self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model and shape a dense reward function for learning policies offline
    Summary: To develop a model that can execute multiple skills we propose a self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model
    ###
    Text: The recent spike in certified Artificial Intelligence tools for healthcare has renewed the debate around adoption of this technology One thread of such debate concerns Explainable AI and its promise to render AI devices more transparent and trustworthy we introduce a distinction between feature importance of low- and high-level features We argue that for data types where low-level features come endowed with a clear semantics such as tabular data
    Summary: we introduce a distinction between feature importance of low- and high-level features to make AI devices more transparent
    ###
    Text: Privacy assistants help users manage their privacy online Their tasks could vary from detecting privacy violations to recommending sharing actions for content that the user intends to share Recent work on these tasks are promising and show that privacy assistants can successfully tackle them However for such privacy assistants to be employed by users it is important that these assistants can explain their decisions to users Accordingly this paper develops a methodology to create explanations of privacy
    Summary: We develop a methodology to create explanations for the decisions of privacy assistants.
    ###
    Text: Modeling and understanding time remains a challenge in contemporary video understanding models In this paper we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations We establish that six existing video-language models struggle to understand even such simple temporal relations We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch Towards this we propose a temporal adaptation recipe on top of one such model VideoCLIP based on post-pretraining on a small amount of video-text data
    Summary:  We show that video-language models struggle to understand temporal relations we propose a temporal adaptation recipe on top of one such model to equip them with temporal awareness without re-training from scratch
    ###
    Text: Although remarkable progress on the neural table-to-text methods has been made the generalization issues hinder the applicability of these models due to the limited source tables Large-scale pretrained language models sound like a promising solution to tackle such issues However how to effectively bridge the gap between the structured table and the text input by fully leveraging table information to fuel the pretrained model In this paper to implement the table-to-text generation with pretrained language model we propose a table structure understanding and text deliberating approach namely TASD Specifically
    Summary: To use large-scale pretrained language models on neural table-to-text methods we propose a table structure understanding and text deliberating approach namely TASD
    ###
    Text: {}
    Summary:
    """

    return few_shots


def summary_shots_translation():

    few_shots = """
    ###
    Text: We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state 3D model Our network learns to predict the object's pose part segmentation and corresponding motion parameters to reproduce the articulation shown in the input image
    Summary: Wir stellen ein neuronales Netz vor, das lernt, die Position eines Objekts und die entsprechenden Bewegungsparameter aus einem Bild vorherzusagen.
    ###
    Text: Unlike language tasks where the output space is usually limited to a set of tokens the output space of visual tasks is more complicated making it difficult to build a unified visual model for various visual tasks In this paper we seek to unify the output space of visual tasks so that we can also build a unified model for visual tasks
    Summary: Wir versuchen, den Output Space von visuellen Aufgaben zu vereinheitlichen. Der Output Space ist komplexer als bei Sprachaufgaben, da dort der Ausgaberaum begrenzt ist
    ###
    Text: Autoformalization seeks to address this by translating proofs written in natural language into a formal representation that is computer-verifiable via interactive theorem provers In this paper we introduce a semantic parsing approach based on the Universal Transformer architecture that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover
    Summary: Wir stellen einen semantischen Parsing-Ansatz vor, der auf der Universal Transformer-Architektur basiert, um in natürlicher Sprache geschriebene Beweise in eine formale Darstellung zu übersetzen
    ###
    Text: In this paper we explored the use of deep learning for the prediction of aortic flow metrics obtained using 4D flow MRI using wearable seismocardiography devices We hypothesized that deep learning could be used to identify pathological changes in blood flow
    Summary: Wir untersuchten den Einsatz von Deep Learning für die Vorhersage von Aortenflussmetriken, die mithilfe von 4D-Flow-MRT unter Verwendung von tragbaren Seismokardiographiegeräten gewonnen wurden
    ###
    Text: Fashion-image editing represents a challenging computer vision task where the goal is to incorporate selected apparel into a given input image Most existing techniques known as Virtual Try-On methods deal with this task by first selecting an example image of the desired apparel and then transferring the clothing onto the target person Conversely in this paper we consider editing fashion images with text descriptions
    Summary: In diesem Artikel beschäftigen wir uns mit der Erstellung von Modebildern mit Hilfe von Textbeschreibungen statt der üblichen Auswahl der Wunschkleidung anhand von Bildern
    ###
    Text: Developing agents that can execute multiple skills by learning from pre-collected datasets is an important problem in robotics where online interaction with the environment is extremely time-consuming In this work we propose a novel self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model and shape a dense reward function for learning policies offline
    Summary: Um ein Agenten-Modell zu optimieren, das mehrere Fertigkeiten von vorhandenen Datensätzen lernen kann, schlagen wir eine selbstüberwachte Lernphase vor, um die Struktur und die Dynamik des Modells zu verstehen
    ###
    Text: The recent spike in certified Artificial Intelligence tools for healthcare has renewed the debate around adoption of this technology One thread of such debate concerns Explainable AI and its promise to render AI devices more transparent and trustworthy we introduce a distinction between feature importance of low- and high-level features We argue that for data types where low-level features come endowed with a clear semantics such as tabular data
    Summary: Wir führen eine Unterscheidung zwischen der Bedeutung von Merkmalen auf niedriger und hoher Ebene in Gesundheitsanwendungen ein, um KI-Tools im Gesundheitswesen transparenter zu machen
    ###
    Text: Privacy assistants help users manage their privacy online Their tasks could vary from detecting privacy violations to recommending sharing actions for content that the user intends to share Recent work on these tasks are promising and show that privacy assistants can successfully tackle them However for such privacy assistants to be employed by users it is important that these assistants can explain their decisions to users Accordingly this paper develops a methodology to create explanations of privacy
    Summary: Wir entwickelten eine Methodik, um Erklärungen für die Entscheidungen von Datenschutzassistenten zu liefern.
    ###
    Text: Modeling and understanding time remains a challenge in contemporary video understanding models In this paper we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations We establish that six existing video-language models struggle to understand even such simple temporal relations We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch Towards this we propose a temporal adaptation recipe on top of one such model VideoCLIP based on post-pretraining on a small amount of video-text data
    Summary: Wir zeigen, dass videosprachliche Modelle Schwierigkeiten haben, zeitliche Beziehungen zu verstehen. Wir schlagen eine Anpassung auf einem solchen Modell vor, um sie mit zeitlichem Bewusstsein auszustatten, ohne sie von Grund auf neu trainieren zu müssen.
    ###
    Text: Although remarkable progress on the neural table-to-text methods has been made the generalization issues hinder the applicability of these models due to the limited source tables Large-scale pretrained language models sound like a promising solution to tackle such issues However how to effectively bridge the gap between the structured table and the text input by fully leveraging table information to fuel the pretrained model In this paper to implement the table-to-text generation with pretrained language model we propose a table structure understanding and text deliberating approach namely TASD Specifically
    Summary: Für die Verwendung umfangreicher vortrainierter Sprachmodelle für neuronale Tabelle-zu-Text-Methoden schlagen wir einen Ansatz für das Verstehen von Tabellenstrukturen und Erkennen von Texten vor, nämlich TASD
    ###
    Text: {}
    Summary:
    """

    return few_shots

def summary_shots_translation_shorter():

    few_shots = """
    ###
    Text: We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state 3D model Our network learns to predict the object's pose part segmentation and corresponding motion parameters to reproduce the articulation shown in the input image
    Summary: Wir stellen ein neuronales Netz vor, das lernt, die Position eines Objekts und die entsprechenden Bewegungsparameter aus einem Bild vorherzusagen.
    ###
    Text: {}
    Summary:
    """

    return few_shots

def summary_shots_translation_tiny():

    few_shots = """
    ###
    Text: That is a text in English.
    Summary: Das ist ein Text auf Deutsch.
    ###
    Text: {}
    Summary:
    """

    return few_shots

def summary_three_shots():
    few_shots =  """This system summaries abstracts of scientific papers.
    ###
    Text: Nowadays, we are witnessing an increasing effort to improve the performance and trustworthiness of Deep Neural Networks (DNNs), with the aim to enable their adoption in safety critical systems such as self-driving cars. Multiple testing techniques are proposed to generate test cases that can expose inconsistencies in the behavior of DNN models. These techniques assume implicitly that the training program is bug-free and appropriately configured. However, satisfying this assumption for a novel problem requires significant engineering work to prepare the data, design the DNN, implement the training program, and tune the hyperparameters in order to produce the model for which current automated test data generators search for corner-case behaviors. All these model training steps can be error-prone. Therefore, it is crucial to detect and correct errors throughout all the engineering steps of DNN-based software systems and not only on the resulting DNN model. In this paper, we gather a catalog of training issues and based on their symptoms and their effects on the behavior of the training program, we propose practical verification routines to detect the aforementioned issues, automatically, by continuously validating that some important properties of the learning dynamics hold during the training. Then, we design, TheDeepChecker, an end-to-end property-based debugging approach for DNN training programs. We assess the effectiveness of TheDeepChecker on synthetic and real-world buggy DL programs and compare it with Amazon SageMaker Debugger (SMD). Results show that TheDeepChecker's on-execution validation of DNN-based program's properties succeeds in revealing several coding bugs and system misconfigurations, early on and at a low cost. Moreover, TheDeepChecker outperforms the SMD's offline rules verification on training logs in terms of detection accuracy and DL bugs coverage.
    Summary: Deep Neural Networks can suffer from errors made in each of its engineering steps, which has not fully been adressed before. We therefore created TheDeepChecker, an end-to-end property based debugging approach for Deep Neural Network training programs to cover all these engineering steps.
    ###
    Text: Federated learning is a training paradigm according to which a server-based model is cooperatively trained using local models running on edge devices and ensuring data privacy. These devices exchange information that induces a substantial communication load, which jeopardises the functioning efficiency. The difficulty of reducing this overhead stands in achieving this without decreasing the model's efficiency (contradictory relation). To do so, many works investigated the compression of the pre/mid/post-trained models and the communication rounds, separately, although they jointly contribute to the communication overload. Our work aims at optimising communication overhead in federated learning by (I) modelling it as a multi-objective problem and (II) applying a multi-objective optimization algorithm (NSGA-II) to solve it. To the best of the author's knowledge, this is the first work that explores the add-in that evolutionary computation could bring for solving such a problem, and considers both the neuron and devices features together. We perform the experimentation by simulating a server/client architecture with 4 slaves. We investigate both convolutional and fully-connected neural networks with 12 and 3 layers, 887,530 and 33,400 weights, respectively. We conducted the validation on the dataset containing 70,000 images. The experiments have shown that our proposal could reduce communication by 99% and maintain an accuracy equal to the one obtained by the FedAvg Algorithm that uses 100% of communications.
    Summary: There is a high communication load in federated learning, an architecture in which a server-based model is cooperatively trained using local models running on edge devices. We were able to reduce the communication overhead significantly relying on the multi-objective optimization algorithm (NSGA-II).
    ###
    Text: Large pretrained (e.g., "foundation") models exhibit distinct capabilities depending on the domain of data they are trained on. While these domains are generic, they may only barely overlap. For example, visual-language models (VLMs) are trained on Internet-scale image captions, but large language models (LMs) are further trained on Internet-scale text with no images (e.g., spreadsheets, SAT questions, code). As a result, these models store different forms of commonsense knowledge across different domains. In this work, we show that this diversity is symbiotic, and can be leveraged through Socratic Models (SMs): a modular framework in which multiple pretrained models may be composed zero-shot i.e., via multimodal-informed prompting, to exchange information with each other and capture new multimodal capabilities, without requiring finetuning. With minimal engineering, SMs are not only competitive with state-of-the-art zero-shot image captioning and video-to-text retrieval, but also enable new applications such as (i) answering free-form questions about egocentric video, (ii) engaging in multimodal assistive dialogue with people (e.g., for cooking recipes) by interfacing with external APIs and databases (e.g., web search), and (iii) robot perception and planning.
    Summary: Large models like visual-language models or large language models are rarely combined. We show that the simple combination of models through socratic models is competitive compared to state-of-the-art image captioning and video-to-text retrieval and also enable new applications.
    ###
    Text: {}
    Summary:
    """

    return few_shots


def translation_prompt():
    few_shots =  """Translate the following sentence from English to German:
    ###
    English: {}
    German:
        """

    return few_shots

def translation_shots():

    few_shots = """
    ###
    English: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration.
    German: Die vorherrschenden Modelle zur Sequenzumsetzung basieren auf komplexen rekurrenten oder konvolutionären neuronalen Netzen in einer Encoder-Decoder-Konfiguration. 
    ###
    English: The best performing models also connect the encoder and decoder through an attention mechanism. 
    German: Die leistungsstärksten Modelle verbinden Kodierer und Dekodierer auch durch einen Aufmerksamkeitsmechanismus.
    ###
    English: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
    German: Wir schlagen eine neue, einfache Netzwerkarchitektur, den Transformer, vor, die ausschließlich auf Aufmerksamkeitsmechanismen basiert und auf Rekursion und Konvolution vollständig verzichtet.
    ###
    English: Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
    German: Experimente mit zwei maschinellen Übersetzungsaufgaben zeigen, dass diese Modelle qualitativ überlegen sind, während sie besser parallelisierbar sind und deutlich weniger Zeit zum Trainieren benötigen. 
    ###
    English: Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU.
    German: Unser Modell erreicht 28,4 BLEU bei der WMT 2014 Englisch-Deutsch-Übersetzungsaufgabe und übertrifft damit die bisher besten Ergebnisse, einschließlich Ensembles, um über 2 BLEU. 
    ###
    English: On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. 
    German: Bei der WMT 2014 Englisch-Französisch-Übersetzungsaufgabe erreicht unser Modell nach 3,5 Tagen Training auf acht GPUs einen neuen BLEU-Wert von 41,8, was nur einen Bruchteil der Trainingskosten der besten Modelle aus der Literatur ausmacht.
    ###
    English: We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
    German: Wir zeigen, dass sich der Transformer gut auf andere Aufgaben verallgemeinern lässt, indem wir ihn erfolgreich auf englisches Conistuency Parsing sowohl mit großen als auch mit begrenzten Trainingsdaten anwenden.
    ###
    English: {}
    German: 
    """

    return few_shots


def keyword_shots():

    few_shots = """
    ###
    Text: We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state 3D model Our network learns to predict the object's pose part segmentation and corresponding motion parameters to reproduce the articulation shown in the input image
    Keywords: object's pose part segmentation, neural network
    ###
    Text: Unlike language tasks where the output space is usually limited to a set of tokens the output space of visual tasks is more complicated making it difficult to build a unified visual model for various visual tasks In this paper we seek to unify the output space of visual tasks so that we can also build a unified model for visual tasks
    Keywords: outspace of visual tasks, unified models
    ###
    Text: Autoformalization seeks to address this by translating proofs written in natural language into a formal representation that is computer-verifiable via interactive theorem provers In this paper we introduce a semantic parsing approach based on the Universal Transformer architecture that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover
    Keywords: proof translation, natural language
    ###
    Text: In this paper we explored the use of deep learning for the prediction of aortic flow metrics obtained using 4D flow MRI using wearable seismocardiography devices We hypothesized that deep learning could be used to identify pathological changes in blood flow
    Keywords: deep learning, aortic flow metrics
    ###
    Text: Fashion-image editing represents a challenging computer vision task where the goal is to incorporate selected apparel into a given input image Most existing techniques known as Virtual Try-On methods deal with this task by first selecting an example image of the desired apparel and then transferring the clothing onto the target person Conversely in this paper we consider editing fashion images with text descriptions
    Keywords: computer vision, fashion-image editing
    ###
    Text: Developing agents that can execute multiple skills by learning from pre-collected datasets is an important problem in robotics where online interaction with the environment is extremely time-consuming In this work we propose a novel self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model and shape a dense reward function for learning policies offline
    Keywords: pre-collected datasets, multiple slills agents
    ###
    Text: The recent spike in certified Artificial Intelligence tools for healthcare has renewed the debate around adoption of this technology One thread of such debate concerns Explainable AI and its promise to render AI devices more transparent and trustworthy we introduce a distinction between feature importance of low- and high-level features We argue that for data types where low-level features come endowed with a clear semantics such as tabular data
    Keywords: explainable AI, feature importance
    ###
    Text: Privacy assistants help users manage their privacy online Their tasks could vary from detecting privacy violations to recommending sharing actions for content that the user intends to share Recent work on these tasks are promising and show that privacy assistants can successfully tackle them However for such privacy assistants to be employed by users it is important that these assistants can explain their decisions to users Accordingly this paper develops a methodology to create explanations of privacy
    Keywords: privacy assistants, decision explanations
    ###
    Text: Modeling and understanding time remains a challenge in contemporary video understanding models In this paper we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations We establish that six existing video-language models struggle to understand even such simple temporal relations We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch Towards this we propose a temporal adaptation recipe on top of one such model VideoCLIP based on post-pretraining on a small amount of video-text data
    Keywords: video understanding, temporal awareness
    ###
    Text: Although remarkable progress on the neural table-to-text methods has been made the generalization issues hinder the applicability of these models due to the limited source tables Large-scale pretrained language models sound like a promising solution to tackle such issues However how to effectively bridge the gap between the structured table and the text input by fully leveraging table information to fuel the pretrained model In this paper to implement the table-to-text generation with pretrained language model we propose a table structure understanding and text deliberating approach namely TASD Specifically
    Keywords: table structure understanding, large-scale language models
    ###
    Text: {}
    Keywords:
    """

    return few_shots

# TO DO
def topic_shots():

    few_shots = """
    ###
    Text: We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state 3D model Our network learns to predict the object's pose part segmentation and corresponding motion parameters to reproduce the articulation shown in the input image
    Topic: 
    ###
    Text: Unlike language tasks where the output space is usually limited to a set of tokens the output space of visual tasks is more complicated making it difficult to build a unified visual model for various visual tasks In this paper we seek to unify the output space of visual tasks so that we can also build a unified model for visual tasks
    Topic: 
    ###
    Text: Autoformalization seeks to address this by translating proofs written in natural language into a formal representation that is computer-verifiable via interactive theorem provers In this paper we introduce a semantic parsing approach based on the Universal Transformer architecture that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover
    Topic: 
    ###
    Text: In this paper we explored the use of deep learning for the prediction of aortic flow metrics obtained using 4D flow MRI using wearable seismocardiography devices We hypothesized that deep learning could be used to identify pathological changes in blood flow
    Topic: 
    ###
    Text: Fashion-image editing represents a challenging computer vision task where the goal is to incorporate selected apparel into a given input image Most existing techniques known as Virtual Try-On methods deal with this task by first selecting an example image of the desired apparel and then transferring the clothing onto the target person Conversely in this paper we consider editing fashion images with text descriptions
    Topic: 
    ###
    Text: Developing agents that can execute multiple skills by learning from pre-collected datasets is an important problem in robotics where online interaction with the environment is extremely time-consuming In this work we propose a novel self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model and shape a dense reward function for learning policies offline
    Topic: 
    ###
    Text: The recent spike in certified Artificial Intelligence tools for healthcare has renewed the debate around adoption of this technology One thread of such debate concerns Explainable AI and its promise to render AI devices more transparent and trustworthy we introduce a distinction between feature importance of low- and high-level features We argue that for data types where low-level features come endowed with a clear semantics such as tabular data
    Topic: 
    ###
    Text: Privacy assistants help users manage their privacy online Their tasks could vary from detecting privacy violations to recommending sharing actions for content that the user intends to share Recent work on these tasks are promising and show that privacy assistants can successfully tackle them However for such privacy assistants to be employed by users it is important that these assistants can explain their decisions to users Accordingly this paper develops a methodology to create explanations of privacy
    Topic: 
    ###
    Text: Modeling and understanding time remains a challenge in contemporary video understanding models In this paper we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations We establish that six existing video-language models struggle to understand even such simple temporal relations We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch Towards this we propose a temporal adaptation recipe on top of one such model VideoCLIP based on post-pretraining on a small amount of video-text data
    Topic: 
    ###
    Text: Although remarkable progress on the neural table-to-text methods has been made the generalization issues hinder the applicability of these models due to the limited source tables Large-scale pretrained language models sound like a promising solution to tackle such issues However how to effectively bridge the gap between the structured table and the text input by fully leveraging table information to fuel the pretrained model In this paper to implement the table-to-text generation with pretrained language model we propose a table structure understanding and text deliberating approach namely TASD Specifically
    Topic: 
    ###
    Text: {}
    Topic:
    """

    return few_shots

def topic_from_keys_shots():

    few_shots = """###
    Keywords: language model data audio speech information approach propose neural
    Topic: natural language processing
    ###
    Keywords: rl learning tasks offline reinforcement policy agents optimization multiagent optimal strategies
    Topic: reinforcement learning
    ###
    Keywords: model training detection data datasets new data model performance
    Topic: data and model performance
    ###
    Keywords: representation data model prediction graph new models tasks properties
    Topic: representation learning
    ###
    Keywords: 3d object objects action recognition detection reconstruction model models learning
    Topic: action reconstruction
    ###
    Keywords: {}
    Topic:
    """

    return few_shots
