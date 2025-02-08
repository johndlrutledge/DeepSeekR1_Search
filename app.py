import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import spaces
from duckduckgo_search import DDGS
import time
import torch
from datetime import datetime
import os
import subprocess
import numpy as np
from gc import collect
from diskcache import Cache
cache = Cache("cache_dir")
if not os.path.exists('cache_dir'): os.makedirs('cache_dir')

torch.cuda.empty_cache()

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

EXAMPLES=["Evaluate the derivitive of x^2 + 2x + 1", "What is the capital of France?", "What is the capital of Germany?", "Why did the allies win WW2?", "Given a society where AI/robots have most of the jobs, what will happen to the displaced workers?", "Will the AI revolution likely lead to a better society or a worse one?", "What are the most common side effects of the COVID-19 vaccine?", "Does evidence indicate that MRNA vaccines exhaust the immune system, leading to a higher risk of other infections?"]


def init_models():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    return model.eval().cuda()

@cache.memoize()
def get_web_results(query, max_results=5):  
    """Get web search results using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{
                "title": result.get("title", ""),
                "snippet": result["body"],
                "url": result["href"],
                "date": result.get("published", "")
            } for result in results]
    except Exception as e:
        return []

def format_prompt(query, context):
    """Format the prompt with web context"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    context_lines = '\n'.join([f'- [{res["title"]}]: {res["snippet"]}' for res in context])
    return f"""{query}

Current Time: {current_time}

Search Results:
{context_lines}

Provide a detailed answer in markdown format.
Answer:"""

def format_sources(web_results):
    """Format sources with more details"""
    if not web_results:
        return "<div class='no-sources'>No sources available</div>"
    
    sources_html = "<div class='sources-container'>"
    for i, res in enumerate(web_results, 1):
        title = res["title"] or "Source"
        date = f"<span class='source-date'>{res['date']}</span>" if res['date'] else ""
        sources_html += f"""
        <div class='source-item'>
            <div class='source-number'>[{i}]</div>
            <div class='source-content'>
                <a href="{res['url']}" target="_blank" class='source-title'>{title}</a>
                {date}
                <div class='source-snippet'>{res['snippet'][:500]}...</div>
            </div>
        </div>
        """
    sources_html += "</div>"
    return sources_html

@cache.memoize()
def generate_answer(prompt):
    """Generate answer using the DeepSeek model"""
    model = init_models()
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=False,
        max_length=32768,
        return_attention_mask=True
    ).to(model.device)
    tokens_left = (32768 - inputs.input_ids.shape[1])//1.5
    answers = []
    for _ in range(3):
        answers.append(str(tokenizer.decode(model.generate(inputs.input_ids,attention_mask=inputs.attention_mask,max_new_tokens=tokens_left,temperature=0.5,top_p=0.98,pad_token_id=tokenizer.eos_token_id,do_sample=True,early_stopping=True, num_beams=3)[0], skip_special_tokens=True)).split("Answer:")[-1].strip())
        if len(max(answers, key=len)) > 2000:
            break
    outputs = max(answers, key=len)
    del model, inputs
    torch.cuda.empty_cache()
    collect()
    return outputs


def process_query(query, history=None):
    """Process user query with streaming effect"""
    print("Processing query:", query)
    if history is None:
        history = []
    web_results = get_web_results(query)
    sources_html = format_sources(web_results)
    
    current_history = history + [[query, "*Searching...*"]]
    yield {
        answer_output: gr.Markdown("*Searching & Thinking...*"),
        sources_output: gr.HTML(sources_html),
        search_btn: gr.Button("Searching...", interactive=False),
        chat_history_display: current_history,
    }
    
    
    prompt = format_prompt(query, web_results)
    final_answer = generate_answer(prompt)
    
    updated_history = history + [[query, final_answer]]
    yield {
        answer_output: gr.Markdown(final_answer),
        sources_output: gr.HTML(sources_html),
        search_btn: gr.Button("Search", interactive=True),
        chat_history_display: updated_history,
    }


css = """
background-image: url("https://picsum.photos/seed/picsum/200/300");
    background-repeat: no-repeat;
    background-size: cover;
.gradio-container {
    max-width: 1200px !important;
    background-color: #f7f7f8 !important;
}

#header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem 0;
    background: #1a1b1e;
    border-radius: 12px;
    color: white;
}

#header h1 {
    color: white;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

#header h3 {
    color: #a8a9ab;
}

.search-container {
    background: #1a1b1e;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 1rem;
    margin-bottom: 1rem;
}

.search-box {
    padding: 1rem;
    background: #2c2d30;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Style the input textbox */
.search-box input[type="text"] {
    background: #3a3b3e !important;
    border: 1px solid #4a4b4e !important;
    color: white !important;
    border-radius: 8px !important;
}

.search-box input[type="text"]::placeholder {
    color: #a8a9ab !important;
}

/* Style the search button */
.search-box button {
    background: #2563eb !important;
    border: none !important;
}

/* Results area styling */
.results-container {
    background: #2c2d30;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}

.answer-box {
    background: #3a3b3e;
    border-radius: 8px;
    padding: 1.5rem;
    color: white;
    margin-bottom: 1rem;
}

.answer-box p {
    color: #e5e7eb;
    line-height: 1.6;
}

.sources-container {
    margin-top: 1rem;
    background: #2c2d30;
    border-radius: 8px;
    padding: 1rem;
}

.source-item {
    display: flex;
    padding: 12px;
    margin: 8px 0;
    background: #3a3b3e;
    border-radius: 8px;
    transition: all 0.2s;
}

.source-item:hover {
    background: #4a4b4e;
}

.source-number {
    font-weight: bold;
    margin-right: 12px;
    color: #60a5fa;
}

.source-content {
    flex: 1;
}

.source-title {
    color: #60a5fa;
    font-weight: 500;
    text-decoration: none;
    display: block;
    margin-bottom: 4px;
}

.source-date {
    color: #a8a9ab;
    font-size: 0.9em;
    margin-left: 8px;
}

.source-snippet {
    color: #e5e7eb;
    font-size: 0.9em;
    line-height: 1.4;
}

.chat-history {
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    background: #2c2d30;
    border-radius: 8px;
    margin-top: 1rem;
}

.examples-container {
    background: #2c2d30;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}

.examples-container button {
    background: #3a3b3e !important;
    border: 1px solid #4a4b4e !important;
    color: #e5e7eb !important;
}

/* Markdown content styling */
.markdown-content {
    color: #e5e7eb !important;
}

.markdown-content h1, .markdown-content h2, .markdown-content h3 {
    color: white !important;
}

.markdown-content a {
    color: #60a5fa !important;
}

/* Accordion styling */
.accordion {
    background: #2c2d30 !important;
    border-radius: 8px !important;
    margin-top: 1rem !important;
}

.voice-selector {
    margin-top: 1rem;
    background: #2c2d30;
    border-radius: 8px;
    padding: 0.5rem;
}

.voice-selector select {
    background: #3a3b3e !important;
    color: white !important;
    border: 1px solid #4a4b4e !important;
}
background-image: url("https://picsum.photos/seed/picsum/200/300");
    background-repeat: no-repeat;
    background-size: cover;
"""


with gr.Blocks(title="Deepseek R1", css=css, theme="dark", analytics_enabled=False) as demo:
    chat_history = gr.State([])
    
    with gr.Column(elem_id="header"):
        gr.Markdown("# 🔍 Search the internet and get answers using a distilled version of Deepseek R1")
    
    with gr.Column(elem_classes="search-container"):
        with gr.Row(elem_classes="search-box"):
            search_input = gr.Textbox(
                label="", 
                placeholder="Ask anything...", 
                scale=5,
                container=False
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)
        
        with gr.Row(elem_classes="results-container"):
            with gr.Column(scale=2):
                with gr.Column(elem_classes="answer-box"):
                    answer_output = gr.Markdown(elem_classes="markdown-content")
                with gr.Accordion("Chat History", open=False, elem_classes="accordion"):
                    chat_history_display = gr.Chatbot(elem_classes="chat-history")
            with gr.Column(scale=1):
                with gr.Column(elem_classes="sources-box"):
                    gr.Markdown("### Sources")
                    sources_output = gr.HTML()
        
        with gr.Row(elem_classes="examples-container"):
            gr.Examples(
                examples = EXAMPLES,
                inputs=search_input,
                label="Examples",
                cache_examples=True,
                examples_per_page=len(EXAMPLES),
                cache_mode="eager",
                run_on_click=True,
                fn=process_query,
                outputs=[answer_output, sources_output, search_btn, chat_history_display]
            )

    
    search_btn.click(
        fn=process_query,
        inputs=[search_input, chat_history],
        outputs=[answer_output, sources_output, search_btn, chat_history_display]
    )
    
    
    search_input.submit(
        fn=process_query,
        inputs=[search_input, chat_history],
        outputs=[answer_output, sources_output, search_btn, chat_history_display]
    )

if __name__ == "__main__":
    
    demo.queue(max_size=1).launch(share=False, inbrowser=True)