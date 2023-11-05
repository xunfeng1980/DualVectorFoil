import uuid

import gradio as gr

import util

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chat_history):
        origin_msg = util.gen_llm_resp(message)
        msgs = str.split(origin_msg,sep='\n\n', maxsplit=45)
        bot_message = []
        for m in msgs:
            full_s = message + m
            if str.strip(m) != "":
                bot_message.append(util.gen_llm_resp(m))
                en_m = util.gen_llm_resp(f"将下面的文字翻译为英文，只返回翻译内容:{full_s}")
                bot_message.append((util.gen_image(en_m), (str(uuid.UUID.bytes))))
        for i, e in enumerate(bot_message):
            if i == 0:
                chat_history.append((message, e))
            else:
                chat_history.append((None, e))
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(server_name='gpu', share=False)
