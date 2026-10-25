import gradio as gr
import torch
import torch.nn as nn
import py_vncorenlp
import os
import unicodedata
from transformers import AutoTokenizer, T5EncoderModel
from model_classes import MultiTaskPhoBERT, MultiTaskBARTpho, XLMRobertaMultiTaskSMART


class MultiTaskViT5(nn.Module):
    def __init__(self, n_sentiment=4, n_topic=10):
        super(MultiTaskViT5, self).__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained("VietAI/vit5-base")
        
        self.t5_encoder.gradient_checkpointing_enable() 
        
        hidden_size = self.t5_encoder.config.d_model
        self.drop = nn.Dropout(p=0.1)
        
        self.sent_out = nn.Linear(hidden_size, n_sentiment)
        self.topic_out = nn.Linear(hidden_size, n_topic)

    def forward(self, input_ids, attention_mask):
        # T5 Encoder trả về last_hidden_state: [Batch, Seq_Len, Hidden]
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        output = self.drop(mean_pooled)
        
        # Lưu ý: output layer tên là sent_out/topic_out (khác fc_s/fc_t của model khác)
        return self.sent_out(output), self.topic_out(output)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

# Lấy đường dẫn gốc tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
RDR_PATH = os.path.join(BASE_DIR, 'vncorenlp')

print(f"👉 Thư mục gốc: {BASE_DIR}")
print(f"👉 Thư mục Weights: {WEIGHTS_DIR}")
print(f"👉 Thư mục VnCoreNLP: {RDR_PATH}")

# Mapping nhãn
ID2SENTIMENT = {
    0: "Neutral (Trung tính)", 
    1: "Positive (Tích cực)", 
    2: "Negative (Tiêu cực)", 
    3: "Toxic (Độc hại)"
}

ID2TOPIC = {
    0: "Spam", 
    1: "News (Tin tức)", 
    2: "Academic (Học tập)", 
    3: "Other (Khác)", 
    4: "Service (Dịch vụ)",
    5: "Jobs (Tuyển dụng)", 
    6: "Personal (Tâm sự)", 
    7: "Social (Xã hội)",
    8: "Help (Hỏi đáp)", 
    9: "Events (Sự kiện)"
}

if not os.path.exists(RDR_PATH):
    print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {RDR_PATH}")

try:
    rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=RDR_PATH, annotators=["wseg"])
    print("✅ VnCoreNLP Loaded Success")
except Exception as e:
    print(f"⚠️ Lỗi khởi tạo VnCoreNLP: {e}")
    rdrsegmenter = None 

def segment_text(text):
    if rdrsegmenter is None: return text
    try:
        sentences = rdrsegmenter.word_segment(text)
        return " ".join(sentences)
    except:
        return text

def smart_load_weights(model, filename, device):
    path = os.path.join(WEIGHTS_DIR, filename)
    
    if not os.path.exists(path):
        print(f"⚠️ Không tìm thấy file: {path}")
        return False
    
    try:
        print(f"⏳ Đang load: {filename}...")
        checkpoint = torch.load(path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f"   ℹ️ Phát hiện Full Checkpoint -> Đang trích xuất model_state_dict...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"   ℹ️ Phát hiện Weights Dictionary -> Load trực tiếp...")
            model.load_state_dict(checkpoint)
            
        model.to(device).eval()
        print(f"   ✅ Load thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi load {filename}:\n   {e}")
        return False


print("\n--- BẮT ĐẦU KHỞI TẠO CÁC MODEL ---")

# --- 1. PhoBERT ---
tokenizer_pho = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model_pho = MultiTaskPhoBERT(4, 10)
if smart_load_weights(model_pho, "phobert_best.pth", DEVICE):
    print("-> PhoBERT sẵn sàng.")

# --- 2. Bartpho ---
tokenizer_bart = AutoTokenizer.from_pretrained("vinai/bartpho-word")
model_bart = MultiTaskBARTpho(4, 10)
if smart_load_weights(model_bart, "bartpho_best.pth", DEVICE):
    print("-> Bartpho sẵn sàng.")

# --- 3. XLM-R ---
tokenizer_xlm = AutoTokenizer.from_pretrained("xlm-roberta-base")
model_xlm = XLMRobertaMultiTaskSMART(4, 10)
if smart_load_weights(model_xlm, "xlmr_best.pth", DEVICE):
    print("-> XLM-R sẵn sàng.")

# --- 4. ViT5 (MỚI) ---
tokenizer_vit5 = AutoTokenizer.from_pretrained("VietAI/vit5-base")
model_vit5 = MultiTaskViT5(4, 10)
if smart_load_weights(model_vit5, "vit5_last.pth", DEVICE): 
    print("-> ViT5 sẵn sàng.")


def predict(text, model_name):
    if not text.strip(): return "", {}, {}
    
    # Chuẩn hóa Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Routing model
    if "PhoBERT" in model_name:
        model = model_pho
        tokenizer = tokenizer_pho
        processed_text = segment_text(text)
        max_len = 256
    elif "Bartpho" in model_name:
        model = model_bart
        tokenizer = tokenizer_bart
        processed_text = segment_text(text)
        max_len = 1024
    elif "XLM-RoBERTa" in model_name:
        model = model_xlm
        tokenizer = tokenizer_xlm
        processed_text = text # XLM-R không cần tách từ
        max_len = 256
    elif "ViT5" in model_name: # Cấu hình cho ViT5
        model = model_vit5
        tokenizer = tokenizer_vit5
        processed_text = segment_text(text) # ViT5 cần tách từ
        max_len = 512
    else:
        return "Lỗi: Model chưa được chọn", {}, {}

    # Inference
    try:
        encoding = tokenizer.encode_plus(
            processed_text, max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            l_s, l_t = model(input_ids, mask)
            p_s = torch.softmax(l_s, dim=1)[0]
            p_t = torch.softmax(l_t, dim=1)[0]

        return processed_text, \
               {ID2SENTIMENT[i]: float(p_s[i]) for i in range(4)}, \
               {ID2TOPIC[i]: float(p_t[i]) for i in range(10)}
    except Exception as e:
        return f"Error Inference: {e}", {}, {}


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"]
)

with gr.Blocks(theme=theme, title="NLP Demo System") as demo:
    
    # --- Header ---
    with gr.Row():
        gr.Markdown(
            """
            # 🏫 NEU Student Confessions Analyzer
            ### 🤖 Hệ thống phân loại Cảm xúc & Chủ đề đa nhiệm (4 Models)
            """
        )
    
    # --- Main Area ---
    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(
                label="Nội dung Confession", 
                lines=8, 
                placeholder="Nhập nội dung bài đăng vào đây...",
                elem_id="input_box"
            )
            
            with gr.Row():
                mod = gr.Dropdown(
                    [
                        "PhoBERT-base-v2", 
                        "Bartpho-word", 
                        "XLM-RoBERTa SMART",
                        "ViT5-base  (Best Acc)"
                    ], 
                    value="PhoBERT-base-v2", 
                    label="Chọn Mô Hình AI",
                    interactive=True
                )
                
            btn = gr.Button("🔍 Phân Tích Ngay", variant="primary", size="lg")
            
            with gr.Accordion("ℹ️ Chi tiết kỹ thuật (Debug)", open=False):
                out_txt = gr.Textbox(label="Văn bản sau tiền xử lý (Segmentation)", lines=3)

        with gr.Column(scale=3):
            gr.Markdown("### 📊 Kết quả Phân tích")
            
            with gr.Group():
                gr.Markdown("**Cảm xúc (Sentiment)**")
                out_s = gr.Label(num_top_classes=4, label=None)
            
            with gr.Group():
                gr.Markdown("**Chủ đề (Topic)**")
                out_t = gr.Label(num_top_classes=5, label=None)

    # --- Examples ---
    gr.Markdown("### 📝 Thử nghiệm nhanh")
    gr.Examples(
        examples=[
            ["Vãi cả đái mấy bọn xài tool đăng kí học phần làm cho tao mất slot, làm tao rất đéo hài lòng", "PhoBERT-base-v2"],
            ["Mình cần tìm bạn nữ áo trắng đi xe Vision biển 29 ở nhà xe B1 sáng nay, mình muốn làm quen ạ.", "Bartpho-word"],
            ["Thầy dạy môn đại tuyến tính nhiệt tình lắm, support sinh viên hết mình luôn.", "XLM-RoBERTa SMART"],
            ["Trường mình cơ sở vật chất tốt nhưng học phí hơi cao so với mặt bằng chung.", "ViT5-base  (Best Acc)"]
        ],
        inputs=[inp, mod],
        label="Chọn câu ví dụ:"
    )

    # --- Footer ---
    gr.Markdown("--- \n*Developed with Gradio & PyTorch*")

    btn.click(predict, [inp, mod], [out_txt, out_s, out_t])

if __name__ == "__main__":
    demo.launch()