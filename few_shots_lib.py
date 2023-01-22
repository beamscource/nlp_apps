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

# TO DO
def trans_summary_shots():

    few_shots = """
    ###
    Text: We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state 3D model Our network learns to predict the object's pose part segmentation and corresponding motion parameters to reproduce the articulation shown in the input image
    Summary:
    ###
    Text: Unlike language tasks where the output space is usually limited to a set of tokens the output space of visual tasks is more complicated making it difficult to build a unified visual model for various visual tasks In this paper we seek to unify the output space of visual tasks so that we can also build a unified model for visual tasks
    Summary:
    ###
    Text: Autoformalization seeks to address this by translating proofs written in natural language into a formal representation that is computer-verifiable via interactive theorem provers In this paper we introduce a semantic parsing approach based on the Universal Transformer architecture that translates elementary mathematical proofs into an equivalent formalization in the language of the Coq interactive theorem prover
    Summary:
    ###
    Text: In this paper we explored the use of deep learning for the prediction of aortic flow metrics obtained using 4D flow MRI using wearable seismocardiography devices We hypothesized that deep learning could be used to identify pathological changes in blood flow
    Summary:
    ###
    Text: Fashion-image editing represents a challenging computer vision task where the goal is to incorporate selected apparel into a given input image Most existing techniques known as Virtual Try-On methods deal with this task by first selecting an example image of the desired apparel and then transferring the clothing onto the target person Conversely in this paper we consider editing fashion images with text descriptions
    Summary:
    ###
    Text: Developing agents that can execute multiple skills by learning from pre-collected datasets is an important problem in robotics where online interaction with the environment is extremely time-consuming In this work we propose a novel self-supervised learning phase on the pre-collected dataset to understand the structure and the dynamics of the model and shape a dense reward function for learning policies offline
    Summary:
    ###
    Text: The recent spike in certified Artificial Intelligence tools for healthcare has renewed the debate around adoption of this technology One thread of such debate concerns Explainable AI and its promise to render AI devices more transparent and trustworthy we introduce a distinction between feature importance of low- and high-level features We argue that for data types where low-level features come endowed with a clear semantics such as tabular data
    Summary:
    ###
    Text: Privacy assistants help users manage their privacy online Their tasks could vary from detecting privacy violations to recommending sharing actions for content that the user intends to share Recent work on these tasks are promising and show that privacy assistants can successfully tackle them However for such privacy assistants to be employed by users it is important that these assistants can explain their decisions to users Accordingly this paper develops a methodology to create explanations of privacy
    Summary:
    ###
    Text: Modeling and understanding time remains a challenge in contemporary video understanding models In this paper we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations We establish that six existing video-language models struggle to understand even such simple temporal relations We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch Towards this we propose a temporal adaptation recipe on top of one such model VideoCLIP based on post-pretraining on a small amount of video-text data
    Summary:
    ###
    Text: Although remarkable progress on the neural table-to-text methods has been made the generalization issues hinder the applicability of these models due to the limited source tables Large-scale pretrained language models sound like a promising solution to tackle such issues However how to effectively bridge the gap between the structured table and the text input by fully leveraging table information to fuel the pretrained model In this paper to implement the table-to-text generation with pretrained language model we propose a table structure understanding and text deliberating approach namely TASD Specifically
    Summary:
    ###
    Text: {}
    Summary:
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

    few_shots = """
    ###
    Text: language model data audio speech information approach propose neural
    Topic: natural language processing
    ###
    Text: rl learning tasks search methods offline reinforcement policy agents algorithms optimization multiagent optimal strategies
    Topic: reinforcement learning
    ###
    Text: model training detection data datasets new data model performance
    Topic: data and model performance
    ###
    Text: representation data model prediction graph new models tasks properties
    Topic: representation learning
    ###
    Text: 3d object objects action recognition detection reconstruction model models learning
    Topic: action reconstruction
    ###
    Text: {}
    Topic:
    """

    return few_shots