import json
import os

import requests
import streamlit as st
import transformers
# load environment variables
from dotenv import load_dotenv
from loguru import logger

from app_features import (load_data, generate_prompt_series, validate_token_threshold, convert_seconds, search_result)
from prompt_templates import question_answering_prompt_series
from reranker import ReRanker
from weaviate_interface import WeaviateClient, WhereFilter

load_dotenv('.env', override=True)

## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'
weaviate_api_key = os.environ['WEAVIATE_API_KEY']
weaviate_url = os.environ['WEAVIATE_ENDPOINT']
hf_token = os.environ['HUGGING_FACE_API_KEY']

### Main definitions
reranker_class_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
llm_available_models = ['meta-llama/Llama-2-13b-chat-hf']
llm_model_name = llm_available_models[0]
vector_db_index_name = "ImpactTheoryMinilm256"
token_threshold = 8000 if llm_model_name == llm_available_models[0] else 3500

## RETRIEVER
client = WeaviateClient(weaviate_api_key, weaviate_url)
logger.info(f"client is live: {client.is_live()}, client is ready: {client.is_ready()}")
available_classes = sorted(client.show_classes())
logger.info(available_classes)

## RERANKER

reranker = ReRanker(model_name=reranker_class_name)

## LLM

## ENCODING

encoding = transformers.AutoTokenizer.from_pretrained(llm_model_name, token=hf_token)

##############
#  END CODE  #
##############
data = load_data(data_path)
# creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))


def call_llm(prompt, temperature=0.1, max_tokens=400):
    hf_token = os.environ['HUGGING_FACE_API_KEY']
    hf_endpoint = os.environ['HUGGING_FACE_ENDPOINT']

    headers = {"Authorization": f"Bearer {hf_token}",
               "Content-Type": "application/json", }

    json_body = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "repetition_penalty": 1.0, "temperature": temperature}
    }

    data = json.dumps(json_body)

    response = requests.request("POST", hf_endpoint, headers=headers, data=data)
    try:
        return json.loads(response.content.decode("utf-8"))
    except:
        return response


def main():
    with st.sidebar:
        guest_input = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

        alpha_input = st.slider('Alpha for Hybrid Search', 0.00, 1.00, step=0.45)
        retrieval_limit = st.slider('Limit for retrieval results', 1, 100, 10)
        reranker_topk = st.slider('Top K for Reranker', 1, 50, 3)
        temperature_input = st.slider('Temperature for LLM', 0.0, 2.0, 1.0)
        class_name = st.selectbox('Class Name:', options=available_classes, index=None, placeholder='Select Class Name')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7, 3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############

            guest_filter = WhereFilter(path=['guest'], operator='Equal',
                                       valueText=guest_input).todict() if guest_input else None

            # make hybrid call to weaviate
            hybrid_response = hybrid_response = client.hybrid_search(query,
                                                                     class_name=class_name,
                                                                     alpha=alpha_input,
                                                                     display_properties=client.display_properties,
                                                                     where_filter=guest_filter,
                                                                     limit=retrieval_limit)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response,
                                              query,
                                              apply_sigmoid=True,
                                              top_k=reranker_topk)

            # validate token count is below threshold

            valid_response = validate_token_threshold(ranked_response,
                                                      question_answering_prompt_series,
                                                      query=query,
                                                      tokenizer=encoding,  # variable from ENCODING,
                                                      token_threshold=token_threshold,
                                                      verbose=True)
            ##############
            #  END CODE  #
            ##############

            prompt = generate_prompt_series(query=query, results=valid_response)

            make_llm_call = True

            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                # creates container for LLM response
                chat_container, response_box = [], st.empty()

                # execute chat call to LLM
                ##############
                # START CODE #
                ##############
                if make_llm_call:
                    for resp in call_llm(prompt=prompt,
                                         temperature=temperature_input,
                                         max_tokens=350):
                        try:
                            with response_box:
                                content = resp.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.write(f'{result}')
                        except Exception as e:
                            print(e)
                            continue
                            ##############
                            #  END CODE  #
                            ##############
            ##############
            # START CODE #
            ##############
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']  # get thumbnail_url
                episode_url = hit['episode_url']  # get episode_url
                title = hit['title']  # get title
                show_length = hit['length']  # get length
                time_string = convert_seconds(show_length)  # convert show_length to readable time string
                ##############
                #  END CODE  #
                ##############
                with col1:
                    st.write(search_result(i=i,
                                           url=episode_url,
                                           guest=hit['guest'],
                                           title=title,
                                           content=hit['content'],
                                           length=time_string),
                             unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    # st.write(f"<a href={episode_url} <img src={image} width='200'></a>",
                    #             unsafe_allow_html=True)
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)


if __name__ == '__main__':
    main()
