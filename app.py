import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Moses-Improved-SMT",
    page_icon=":star:",
)

hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown('''
    # Neural Network based Optimized Pruning Strategy for Statistical Machine Translation
    ***
    ''')

options = st.sidebar.selectbox(
            'Choose Model: ',
            ('Default setting', 'Logistic_Regression', 'XGBoost', 'Manual', 'Compare all models')
     )

st.markdown(f'## Model selected: ```{options}```')


if(options == "Manual"):
    st.sidebar.write("Set beam-threshold:")
    beam_input = st.sidebar.number_input("Enter custom beam-threshold")


if(options == "Manual"):
    st.sidebar.write("Set stack size:")
    stack_input = st.sidebar.number_input("Enter custom stack size")


show_docs = st.sidebar.checkbox("Show Docs")
if(show_docs):
    st.sidebar.markdown(
        '''
        #### Default setting :
        * Runs the moses decoder only without any model predicting beam threshold and stack size.
        * Uses Moses decoder's default beam threshold and stack size.
        #### Logistic_Regression :
        * Runs the moses decoder with a logistic regression model to predict appropriate stack size, depending on the input text.
        #### XGBoost :
        * Runs the moses decoder with a XGBoost model to predict appropriate stack size, depending on the input text.
        #### Manual :
        * Runs the moses decoder with custom beam threshold and custom stack size which can be tweaked by the user.
        #### Compare all models :
        * Runs the program with Default setting, Logistic_Regression and XGBoost and also compares three of them graphically.
        '''
    )


txt = st.text_area('Text to analyze', '''''')

translate = st.button('Translate')




# if translate:
#     try:
#         with st.spinner('Translating...'):
#             decoder = generate_output()
            
#             if(options == "Manual"):
#                 result_dict = decoder.output(options, txt, beam_threshold_size = beam_input, stack_sz= stack_input)
            
#             elif options == "Compare all models":
#                 all_result = {}

#                 result_dict1 = decoder.output("Default setting", txt)
#                 if(result_dict1):
#                     st.markdown('#### for Default setting: ')
#                     st.markdown("- Translated text: " + str(result_dict1["translation"]))
#                     st.markdown("- Decoding time: " + str(result_dict1["decoding_time"]) + " seconds.\n")
                
#                 with st.spinner("Running the rest of the models..."):
#                     result_dict2 = decoder.output("Logistic_Regression", txt)

#                     st.markdown('#### for Logistic_Regression: ')
#                     st.markdown("- Translated text: " + str(result_dict2["translation"]))
#                     st.markdown("- Decoding time: " + str(result_dict2["decoding_time"]) + " seconds.\n")

#                 with st.spinner("Running the rest of the models..."):
#                     result_dict3 = decoder.output("XGBoost", txt)

#                     st.markdown(f'#### for XGBoost: ')
#                     st.markdown("- Translated text: " + str(result_dict3["translation"]))
#                     st.markdown("- Decoding time: " + str(result_dict3["decoding_time"]) + " seconds.\n")

#                 all_result["Default setting"] = result_dict1
#                 all_result["Logistic_Regression"] = result_dict2
#                 all_result["XGBoost"] = result_dict3

#                 result_dict = all_result

#                 y_axis = [
#                     all_result["Default setting"]["decoding_time"], 
#                     all_result["Logistic_Regression"]["decoding_time"], 
#                     all_result["XGBoost"]["decoding_time"]
#                     ]

#                 x_axis = [
#                     "Default setting", 
#                     "Logistic_Regression", 
#                     "XGBoost"
#                     ]
                
#                 fig, ax = plt.subplots()
#                 plt.xlabel("Model Type")
#                 plt.ylabel("Decoding Time")
#                 ax.plot(x_axis, y_axis)
                
#                 st.pyplot(fig)

#             else: 
#                 result_dict = decoder.output(options, txt)
        
        
#         if(options != "Compare all models"):
#             st.markdown("#### Translated text: " + str(result_dict["translation"]))
#             st.markdown("#### Decoding time: " + str(result_dict["decoding_time"]) + " seconds")
    
#     except:
#         st.write("An unknown error occurred.")