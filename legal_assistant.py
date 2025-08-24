import asyncio
import gradio as gr

from services.legal_generator import (
    generate_legal_document,
    normalize_doc_type,
    stream_legal_document_sync,
)
from services.export_utils import export_docx, export_pdf


DOC_OPTIONS = [
    "Rental Agreement",
    "Employment Contract",
    "Business Partnership Agreement",
    "NDA",
]


def _should_show_duration(doc_type_label: str) -> bool:
    key = (doc_type_label or "").strip().lower()
    canonical = normalize_doc_type(key)
    return canonical == "rental agreement"


def _should_show_salary(doc_type_label: str) -> bool:
    key = (doc_type_label or "").strip().lower()
    canonical = normalize_doc_type(key)
    return canonical == "employment contract"


def _normalize_label_to_key(label: str) -> str:
    return (label or "").strip().lower()


def sync_generate(doc_type_label: str, party1: str, party2: str, duration: str, salary: str,
                  temperature: float, top_p: float, num_predict: int, stream: bool) -> str:
    try:
        doc_type_key = _normalize_label_to_key(doc_type_label)
        if stream:
            # yield progressively for Gradio streaming support
            accumulated = ""
            for chunk in stream_legal_document_sync(
                doc_type=doc_type_key,
                party1=party1,
                party2=party2,
                duration=duration,
                salary=salary,
                temperature=temperature,
                top_p=top_p,
                num_predict=num_predict,
            ):
                accumulated += chunk
                yield accumulated
            return
        else:
            # Run the async generator within Gradio's sync fn
            return asyncio.run(
                generate_legal_document(
                    doc_type=doc_type_key,
                    party1=party1,
                    party2=party2,
                    duration=duration,
                    salary=salary,
                    temperature=temperature,
                    top_p=top_p,
                    num_predict=num_predict,
                )
            )
    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        return f"Generation error: {e}"


def build_interface() -> gr.Blocks:
    custom_css = """
    /* Make Radio options display on a single horizontal row with scroll if overflow */
    .inline-radio .gr-radio { display: flex; flex-direction: row; gap: 10px; flex-wrap: nowrap; overflow-x: auto; }
    .inline-radio .gr-radio > label { display: inline-flex; align-items: center; margin-right: 4px; white-space: nowrap; }
    /* Back-compat for older markup that used .wrap */
    .inline-radio .wrap { display: flex; flex-direction: row !important; gap: 10px; flex-wrap: nowrap; overflow-x: auto; }
    """
    with gr.Blocks(title="AI-Powered Legal Assistant", css=custom_css) as demo:
        gr.Markdown("# AI-Powered Legal Assistant\nSelect a document type, enter party names, and generate a professional legal contract.")

        # Row 1: Label on the left, Stream toggle on the right
        with gr.Row():
            with gr.Column(scale=9, min_width=200):
                gr.Markdown("**Document Type**")
            with gr.Column(scale=3, min_width=180):
                stream_chk = gr.Checkbox(False, label="Stream output")

        # Row 2: Full-width horizontal choices
        with gr.Row():
            doc_type = gr.Radio(
                DOC_OPTIONS,
                label=None,
                value=DOC_OPTIONS[0],
                elem_classes=["inline-radio"],
                show_label=False,
            )

        with gr.Row():
            with gr.Accordion("Quality", open=False):
                temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            with gr.Accordion("Length", open=False):
                num_predict = gr.Slider(64, 4096, value=512, step=64, label="Max tokens")

        with gr.Row():
            party1 = gr.Textbox(label="Party 1 Name")
            party2 = gr.Textbox(label="Party 2 Name")
            duration = gr.Textbox(label="Duration (months)", placeholder="e.g., 12", visible=True)
            salary = gr.Textbox(label="Salary (per year)", placeholder="e.g., $50,000", visible=False)

        def _toggle_fields(selected):
            return gr.update(visible=_should_show_duration(selected)), gr.update(visible=_should_show_salary(selected))

        doc_type.change(_toggle_fields, inputs=[doc_type], outputs=[duration, salary])

        generate_btn = gr.Button("Generate Document")
        output = gr.Textbox(label="Generated Legal Document")
        with gr.Row():
            download_docx = gr.Button("Download DOCX")
            download_pdf = gr.Button("Download PDF")
        file_docx = gr.File(label="DOCX File", visible=False)
        file_pdf = gr.File(label="PDF File", visible=False)

        generate_btn.click(
            fn=sync_generate,
            inputs=[doc_type, party1, party2, duration, salary, temperature, top_p, num_predict, stream_chk],
            outputs=[output],
            api_name="generate",
        )

        def _do_export_docx(text: str):
            if not text:
                return gr.update(visible=False), None
            path = export_docx(text)
            return gr.update(visible=True, value=path), path

        def _do_export_pdf(text: str):
            if not text:
                return gr.update(visible=False), None
            path = export_pdf(text)
            return gr.update(visible=True, value=path), path

        download_docx.click(_do_export_docx, inputs=[output], outputs=[file_docx])
        download_pdf.click(_do_export_pdf, inputs=[output], outputs=[file_pdf])

    return demo


# Create Gradio interface (used standalone and can be mounted by FastAPI)
interface = build_interface()

# Launch standalone only when executed directly
if __name__ == "__main__":
    interface.launch()



# # CLI test (optional)
# if __name__ == "__main__":
#     print("### AI-Generated Contract ###")
#     print(asyncio.run(generate_legal_document(doc_type="rental agreement", party1="John Doe", party2="Jane Smith", duration="12")))



