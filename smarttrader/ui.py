import gradio as gr
import pandas as pd
from datetime import datetime
from inference import run_inference
from strategy import strategy

df_css = """
table { overflow: hidden;}
"""

def process_and_display(date):
    predictions = run_inference(date)
    print(predictions)
    # highest_price = predictions.max().max()
    # lowest_price = predictions.min().min()
    # avg_close_price = predictions['Close'].mean()
    highest_price = predictions['High'].max()
    lowest_price = predictions['Low'].min()
    avg_close_price = predictions['Close'].mean()
    
    strategy_df = strategy(predictions)
    # print(strategy_df)
    
    stats_df = pd.DataFrame({
        'Metric': ['Highest Price', 'Lowest Price', 'Avg Close Price'],
        'Value': [f'${highest_price:.2f}', f'${lowest_price:.2f}', f'${avg_close_price:.2f}']
    })
    # print(strategy_df)
    strategy_df['Date'] = strategy_df['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    strategy_df.set_index('Date', inplace=True)
    
    predictions['Date'] = predictions['Date'].apply(lambda x: x.strftime('%m-%d-%Y'))
    return stats_df.values.tolist(), strategy_df.reset_index().values.tolist(), predictions.round(2).values.tolist()[1:]


# Create the Gradio interface
with gr.Blocks(title="SmartTrader", css=df_css) as demo:
    gr.Markdown("# Stock Price Prediction and Trading Strategy")
    
    with gr.Row():
        date_input = gr.Textbox(
            label="Date",
            value=datetime.today().strftime('%Y-%m-%d'),
            placeholder="YYYY-MM-DD"
        )
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Predicted prices for the next five business days")
            table1 = gr.Dataframe(
                headers=['Metric', 'Value'],
                row_count=3,
                col_count=2
            )
        
        with gr.Column():
            gr.Markdown("### Recommended Trading Strategy")
            table2 = gr.Dataframe(
                headers=['Date', 'Action'],
                row_count=5,
                col_count=2
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Detailed Price Predictions")
            table3 = gr.Dataframe(
                headers=['Date', 'Open', 'High', 'Low', 'Close'],
                row_count=5,
                col_count=5
            )
    
    submit_btn.click(
        fn=process_and_display,
        inputs=[date_input],
        outputs=[table1, table2, table3]
    )

def main():
    demo.launch(server_name="0.0.0.0", server_port=8080)


if __name__ == "__main__":
    main()    