import gradio as gr
from datetime import datetime
from inference import predict_stocks

df_css = """
table { overflow: hidden;}
"""

def build_ui():
    with gr.Blocks(title="SmartTrader - Stock Market Prediction", css=df_css) as demo:
        gr.Markdown("# SmartTrader: NVDA & NVDQ Trading Assistant")
        gr.Markdown("### Predict stock prices and get trading recommendations")
        
        with gr.Row():
            date_input = gr.Textbox(
                label="Select Date (YYYY-MM-DD)",
                value=datetime.now().strftime('%Y-%m-%d')
            )
            predict_btn = gr.Button("Predict")
        
        with gr.Row():
            plot_output = gr.Plot(label="Stock Analysis")
        
        with gr.Row():
            table_output = gr.Dataframe(
                headers=["Date", "Open", "High", "Low", "Close", "Action"],
                label="Predictions and Trading Recommendations for $NVDA",
                wrap=True,
                row_count=(5, "fixed"),
                col_count=(6, "fixed"),
                elem_classes=["df"],
            )
        
        predict_btn.click(
            predict_stocks,
            inputs=[date_input],
            outputs=[plot_output, table_output]
        )
        
        with gr.Row():
            gr.Markdown("""
            ### Features:
            - 5-day price predictions for NVDA
            - Trading Strategy Generation 
            - Interactive price charts
            """)

            gr.Markdown("""
            ### Submitted by:
            - Ishan Sharma
            - Manav Goel
            - Ritesh Singh
            """)
        
        return demo

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
