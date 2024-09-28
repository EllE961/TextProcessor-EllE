#  ████████      ██          ██          ████████ 
#  ██            ██          ██          ██       
#  ██  ████      ██          ██          ██  ████
#  ██            ██          ██          ██       
#  ████████      ████████    ████████    ████████ 

import torch
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from tkinter import ttk  # For enhanced widgets
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import threading
import queue

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Define model names
PARAPHRASE_MODEL_NAME = 'Vamsi/T5_Paraphrase_Paws'
SUMMARIZE_MODEL_NAME = 't5-base'  # You can change this to another model if necessary

# Initialize a queue for thread-safe GUI updates
output_queue = queue.Queue()

# Load the paraphrase model and tokenizer
try:
    tokenizer_paraphrase = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL_NAME, legacy=False)
    model_paraphrase = T5ForConditionalGeneration.from_pretrained(PARAPHRASE_MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_paraphrase.to(device)
except Exception as e:
    logger.error(f'Error loading paraphrase model/tokenizer: {e}')
    messagebox.showerror("Initialization Error", f"Failed to load the paraphrase model: {e}")
    exit(1)

# Load the summarization model and tokenizer
try:
    tokenizer_summarize = T5Tokenizer.from_pretrained(SUMMARIZE_MODEL_NAME, legacy=False)
    model_summarize = T5ForConditionalGeneration.from_pretrained(SUMMARIZE_MODEL_NAME)
    model_summarize.to(device)
except Exception as e:
    logger.error(f'Error loading summarization model/tokenizer: {e}')
    messagebox.showerror("Initialization Error", f"Failed to load the summarization model: {e}")
    exit(1)

def paraphrase_text(text, max_length=250, num_return_sequences=3, 
                   temperature=0.7, top_k=50, top_p=0.85):
    """
    Generate paraphrased text that retains more details, focusing on paragraphs.
    """
    try:
        # Tokenize the input text using the updated tokenizer interface
        inputs = tokenizer_paraphrase.encode_plus(
            "paraphrase: " + text,
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=max_length
        ).to(device)

        # Generate paraphrased output
        with torch.no_grad():
            outputs = model_paraphrase.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=10,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True  # Enable sampling to use temperature and top_p
            )

        # Decode the generated outputs
        paraphrased_texts = [
            tokenizer_paraphrase.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for output in outputs
        ]

        return paraphrased_texts

    except Exception as e:
        logger.error(f'Error during paraphrasing: {e}')
        messagebox.showerror("Paraphrasing Error", f"An error occurred during paraphrasing: {e}")
        return []

def summarize_text(text, max_length=150, min_length=60, num_return_sequences=3, temperature=0.7, top_p=0.85):
    """
    Generate summarized text of the input.
    """
    try:
        # Tokenize the input text
        inputs = tokenizer_summarize.encode_plus(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding='longest'
        ).to(device)

        # Generate the summary
        with torch.no_grad():
            outputs = model_summarize.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=50,
                top_p=top_p,
                do_sample=True  # Enable sampling to get diverse outputs
            )

        # Decode the summary output
        summarized_texts = [
            tokenizer_summarize.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for output in outputs
        ]

        return summarized_texts

    except Exception as e:
        logger.error(f'Error during summarization: {e}')
        messagebox.showerror("Summarization Error", f"An error occurred during summarization: {e}")
        return []

def generate_output():
    """Callback for the generate button."""
    input_text = text_entry.get("1.0", "end-1c")  # Get input from text box
    operation = operation_var.get()

    if not input_text.strip():
        messagebox.showwarning("Input Error", "Please enter some text to process.")
        progress_bar.stop()
        progress_bar.pack_forget()
        loading_label.pack_forget()
        generate_button.config(state=tk.NORMAL)
        return

    # Process based on selected operation
    if operation == "Paraphrase":
        output_texts = paraphrase_text(input_text)
    elif operation == "Summarize":
        output_texts = summarize_text(input_text)
    else:
        messagebox.showerror("Operation Error", "Please select a valid operation.")
        progress_bar.stop()
        progress_bar.pack_forget()
        loading_label.pack_forget()
        generate_button.config(state=tk.NORMAL)
        return

    # Put the results in the queue
    output_queue.put((operation, output_texts))

def process_queue():
    """Process items in the queue and update the GUI."""
    try:
        while True:
            operation, output_texts = output_queue.get_nowait()

            # Clear previous results
            output_text.delete("1.0", tk.END)

            # Add outputs to the output area
            if output_texts:
                for idx, output in enumerate(output_texts, 1):
                    output_text.insert(tk.END, f"Output {idx}:\n{output}\n\n")
            else:
                output_text.insert(tk.END, "No results generated.")

            # Re-enable the Generate button and stop/hide the progress bar and loading label
            generate_button.config(state=tk.NORMAL)
            progress_bar.stop()
            progress_bar.pack_forget()
            loading_label.pack_forget()

    except queue.Empty:
        pass
    finally:
        # Check the queue again after 100 ms
        window.after(100, process_queue)

def generate_output_thread():
    """Run the generate_output function in a separate thread to prevent GUI freezing."""
    # Disable the Generate button to prevent multiple clicks
    generate_button.config(state=tk.DISABLED)
    # Show and start the loading label and progress bar
    loading_label.pack(pady=(0, 5))
    progress_bar.pack(pady=(0, 10), fill=tk.X)
    progress_bar.start()
    # Start the thread
    thread = threading.Thread(target=generate_output)
    thread.start()

def copy_to_clipboard():
    """Copy selected output to the clipboard."""
    try:
        selected = output_text.get("sel.first", "sel.last")
        window.clipboard_clear()
        window.clipboard_append(selected)
        messagebox.showinfo("Copied", "Selected text copied to clipboard!")
    except tk.TclError:
        messagebox.showwarning("Selection Error", "Please select an output to copy.")

def clear_fields():
    """Clear all input and output fields."""
    text_entry.delete("1.0", tk.END)
    output_text.delete("1.0", tk.END)

def save_output():
    """Save the output text to a file."""
    try:
        content = output_text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showwarning("Save Error", "There is no content to save.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            messagebox.showinfo("Saved", f"Output saved to {file_path}")
    except Exception as e:
        logger.error(f'Error during saving output: {e}')
        messagebox.showerror("Save Error", f"An error occurred while saving the file: {e}")

# Initialize the main window
window = tk.Tk()
window.title("Text Processor - EllE")
window.geometry("900x800")
window.config(bg="#f5f5f5")

# Create a frame for better layout management
main_frame = ttk.Frame(window, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Create and place widgets
label = ttk.Label(main_frame, text="Enter text to process:", background="#f5f5f5", font=("Arial", 12))
label.pack(pady=(0, 5), anchor='w')

# Text Entry for User Input
text_entry = scrolledtext.ScrolledText(
    main_frame, 
    height=15, 
    width=100, 
    wrap=tk.WORD, 
    font=("Arial", 11)
)
text_entry.pack(pady=(0, 10), fill=tk.BOTH, expand=True)

# Operation Selection Frame
operation_frame = ttk.Frame(main_frame)
operation_frame.pack(pady=(0, 10), anchor='w')

operation_var = tk.StringVar(value="Paraphrase")

paraphrase_radio = ttk.Radiobutton(
    operation_frame, 
    text="Paraphrase", 
    variable=operation_var, 
    value="Paraphrase"
)
paraphrase_radio.pack(side=tk.LEFT, padx=(0, 20))

summarize_radio = ttk.Radiobutton(
    operation_frame, 
    text="Summarize", 
    variable=operation_var, 
    value="Summarize"
)
summarize_radio.pack(side=tk.LEFT)

# Generate Button
generate_button = ttk.Button(main_frame, text="Generate", command=generate_output_thread)
generate_button.pack(pady=(0, 10))

# Loading Label
loading_label = ttk.Label(main_frame, text="Loading...", background="#f5f5f5", font=("Arial", 12), foreground="blue")

# Progress Bar (Initially hidden)
progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')

# Label for Results
result_label = ttk.Label(main_frame, text="Output:", background="#f5f5f5", font=("Arial", 12))
result_label.pack(pady=(10, 5), anchor='w')

# ScrolledText to Display Output Text
output_text = scrolledtext.ScrolledText(
    main_frame, 
    height=25, 
    width=100, 
    wrap=tk.WORD, 
    font=("Arial", 11)
)
output_text.pack(pady=(0, 10), fill=tk.BOTH, expand=True)

# Buttons Frame
buttons_frame = ttk.Frame(main_frame)
buttons_frame.pack(pady=(0, 10))

# Copy to Clipboard Button
copy_button = ttk.Button(buttons_frame, text="Copy Selected Text", command=copy_to_clipboard)
copy_button.pack(side=tk.LEFT, padx=5)

# Save Output Button
save_button = ttk.Button(buttons_frame, text="Save Output", command=save_output)
save_button.pack(side=tk.LEFT, padx=5)

# Clear Button
clear_button = ttk.Button(buttons_frame, text="Clear", command=clear_fields)
clear_button.pack(side=tk.LEFT, padx=5)

# Start processing the queue
window.after(100, process_queue)

# Run the application
window.mainloop()
