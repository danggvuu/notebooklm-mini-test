"""
Streamlit Web UI — Tầng Giao diện
Phân chia theo Workspace, hỗ trợ streaming response, upload status tracking.
"""
import streamlit as st
import httpx
import json
import uuid
from src.config import settings
from src.interfaces.styles import GLOBAL_CSS

_API = settings.api_url


def _get_session_id():
    """Get or create a persistent session ID for this browser session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def _api(method: str, path: str, **kwargs):
    try:
        res = httpx.request(method, f"{_API}{path}", timeout=180.0, **kwargs)
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail", res.text)
            except Exception:
                detail = res.text

            if "API key required" in detail or "API_KEY" in detail or "API key not valid" in detail:
                st.error(
                    "🔑 **Lỗi cấu hình API Key:** Bạn chưa thiết lập hoặc khoá `GOOGLE_API_KEY` trong file `.env` không hợp lệ!\n\n"
                    "**Cách sửa nhanh:**\n"
                    "1. Mở file `.env` ở thư mục dự án.\n"
                    "2. Tìm dòng `GOOGLE_API_KEY=`.\n"
                    "3. Điền API Key Gemini của bạn vào (ví dụ: `GOOGLE_API_KEY=AIzaSy...`).\n"
                    "4. Lưu file lại và thực hiện lại thao tác vừa rồi."
                )
            else:
                st.error(f"❌ **Lỗi từ hệ thống Backend:** {detail}")
            return None
        return res.json()
    except Exception as e:
        st.error(f"Lỗi kết nối đến API backend: {e}")
        return None


def _api_stream(path: str, payload: dict):
    """Call streaming API and yield text chunks."""
    try:
        with httpx.stream("POST", f"{_API}{path}", json=payload, timeout=180.0) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("done"):
                        break
                    if "text" in data:
                        yield data["text"]
    except Exception as e:
        yield f"\n\n⚠️ Lỗi streaming: {e}"


def _sidebar():
    st.sidebar.markdown("<h2 style='font-family:Space Grotesk; font-weight:700; color:#818cf8;'>📚 Quản lý tài liệu</h2>", unsafe_allow_html=True)

    # Workspace indicator
    session_id = _get_session_id()
    st.sidebar.markdown(
        f"<div style='font-size:0.75rem; color:#64748b; margin-bottom:15px;'>🏷️ Workspace: <code>{session_id}</code></div>",
        unsafe_allow_html=True,
    )

    # 1. File Upload (supports multiple formats)
    uploaded_file = st.sidebar.file_uploader(
        "Tải lên tài liệu",
        type=["pdf", "docx", "pptx", "xlsx", "html", "md", "txt"],
    )
    if uploaded_file is not None:
        if st.sidebar.button("Index tài liệu", use_container_width=True):
            with st.spinner("Đang gửi tài liệu để xử lý bất đồng bộ..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                res = _api("POST", "/upload", files=files)
                if res:
                    task_id = res.get("task_id", "")
                    st.session_state["upload_task_id"] = task_id
                    st.sidebar.info(f"📤 Đã gửi! Task ID: `{task_id}`. Đang xử lý nền...")

    # Upload status tracking
    if "upload_task_id" in st.session_state:
        task_id = st.session_state["upload_task_id"]
        status = _api("GET", f"/upload/status/{task_id}")
        if status:
            s = status["status"]
            if s == "done":
                st.sidebar.success(f"✅ Hoàn tất! {status['filename']} — {status['chunks_indexed']} chunks")
                del st.session_state["upload_task_id"]
                if "docs" in st.session_state:
                    del st.session_state["docs"]
            elif s == "error":
                st.sidebar.error(f"❌ Lỗi: {status.get('error_message', 'Unknown')}")
                del st.session_state["upload_task_id"]
            elif s == "processing":
                st.sidebar.warning(f"⏳ Đang xử lý: {status['filename']}...")
            else:
                st.sidebar.info(f"🕐 Chờ xử lý: {status['filename']}...")

    st.sidebar.markdown("---")

    # 2. Document Selection
    docs = _api("GET", "/documents")
    if not docs:
        st.sidebar.warning("Chưa có tài liệu nào trong thư viện. Vui lòng tải lên ở trên.")
        return None, None

    doc_options = ["Toàn bộ tài liệu (Corpus)"] + [d["filename"] for d in docs]
    selected_doc = st.sidebar.selectbox("Chọn tài liệu làm việc", doc_options)

    # Selected document target string
    doc_target = None if selected_doc == "Toàn bộ tài liệu (Corpus)" else selected_doc

    # 3. Optional Page Filters
    page_filter = None
    if doc_target:
        doc_info = next((d for d in docs if d["filename"] == doc_target), None)
        if doc_info and doc_info["num_pages"] > 1:
            page_filter = st.sidebar.slider("Lọc theo Trang (0 = Không lọc)", min_value=0, max_value=doc_info["num_pages"], value=0)
            if page_filter == 0:
                page_filter = None

    # Display statistics
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='font-family:Space Grotesk; font-size:1rem; color:#94a3b8;'>📊 Thống kê thư viện</h3>", unsafe_allow_html=True)
    for d in docs:
        st.sidebar.markdown(
            f"<div style='font-size:0.85rem; color:#64748b; margin-bottom:5px;'>• <b>{d['filename']}</b>: {d['num_pages']} trang, {d['num_chunks']} chunks</div>",
            unsafe_allow_html=True
        )

    return doc_target, page_filter


def _tab_chat(selected_doc, page_filter):
    st.markdown("<h2 style='font-family:Space Grotesk; font-weight:700;'>💬 Hỏi đáp thông minh (RAG)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Đặt câu hỏi và nhận câu trả lời có trích dẫn nguồn chính xác từ tài liệu của bạn. Hỗ trợ streaming real-time.</p>", unsafe_allow_html=True)

    # Chat History Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Streaming toggle
    use_streaming = st.checkbox("🚀 Streaming mode (SSE)", value=True)

    # Clear chat button
    if st.button("Xóa lịch sử chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("📚 Nguồn trích dẫn (Citations)"):
                    for c in msg["citations"]:
                        st.markdown(f"<span class='source-tag'>{c['source_marker']}</span> <b>{c['filename']}</b> (Trang {c['page']})", unsafe_allow_html=True)

    # Chat Input
    query = st.chat_input("Nhập câu hỏi của bạn về tài liệu ở đây...")
    if query:
        # User message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Build filters
        filters = {}
        if page_filter:
            filters["page"] = page_filter
        if selected_doc:
            filters["filename"] = selected_doc

        # Assistant response
        with st.chat_message("assistant"):
            if use_streaming:
                # Streaming mode
                payload = {
                    "question": query,
                    "k": settings.top_k,
                    "filters": filters if filters else None,
                    "session_id": _get_session_id(),
                }
                response_placeholder = st.empty()
                collected = []
                for chunk in _api_stream("/ask/stream", payload):
                    collected.append(chunk)
                    response_placeholder.markdown("".join(collected))

                full_answer = "".join(collected)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "citations": [],
                })
            else:
                # Non-streaming mode
                with st.spinner("Đang suy nghĩ và trích xuất nguồn..."):
                    payload = {
                        "question": query,
                        "k": settings.top_k,
                        "filters": filters if filters else None,
                        "session_id": _get_session_id(),
                    }
                    res = _api("POST", "/ask", json=payload)
                    if res:
                        st.markdown(res["answer"])

                        if res["citations"]:
                            with st.expander("📚 Nguồn trích dẫn (Citations)"):
                                for c in res["citations"]:
                                    st.markdown(f"<span class='source-tag'>{c['source_marker']}</span> <b>{c['filename']}</b> (Trang {c['page']})", unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": res["answer"],
                            "citations": res["citations"]
                        })

        # Feedback buttons
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"fb_up_{len(st.session_state.messages)}"):
                _api("POST", "/feedback", json={"question": query, "feedback_type": "up"})
                st.toast("Cảm ơn phản hồi! 👍")
        with col2:
            if st.button("👎", key=f"fb_down_{len(st.session_state.messages)}"):
                _api("POST", "/feedback", json={"question": query, "feedback_type": "down"})
                st.toast("Cảm ơn phản hồi! Chúng tôi sẽ cải thiện. 🙏")


def _tab_summary(selected_doc, page_filter):
    st.markdown("<h2 style='font-family:Space Grotesk; font-weight:700;'>📝 Tóm tắt & Hệ thống hóa</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Tóm tắt nhanh chóng tài liệu dài sử dụng chiến lược Map-Reduce thông minh.</p>", unsafe_allow_html=True)

    # Custom query guide for summarization
    summary_focus = st.text_input("Trọng tâm tóm tắt (Để trống để tóm tắt toàn bộ tài liệu)", placeholder="Ví dụ: các khái niệm cốt lõi, công thức toán học...")

    if st.button("Bắt đầu tóm tắt"):
        with st.spinner("Đang đọc và xử lý bản tóm tắt..."):
            filters = {}
            if page_filter:
                filters["page"] = page_filter

            payload = {
                "document": selected_doc,
                "query": summary_focus if summary_focus.strip() else None,
                "filters": filters if filters else None
            }
            res = _api("POST", "/summarize", json=payload)
            if res:
                st.session_state["active_summary"] = res

    if "active_summary" in st.session_state:
        sum_data = st.session_state["active_summary"]

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#818cf8; font-family:Space Grotesk;'>✨ Bản tóm tắt chính</h3>", unsafe_allow_html=True)
        st.markdown(sum_data["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#818cf8; font-family:Space Grotesk;'>📌 Các ý chính nổi bật</h3>", unsafe_allow_html=True)
        for kp in sum_data["key_points"]:
            st.markdown(f"- {kp}")
        st.markdown("</div>", unsafe_allow_html=True)

        if sum_data["citations"]:
            with st.expander("📚 Tài liệu tham khảo"):
                for c in sum_data["citations"]:
                    st.markdown(f"<span class='source-tag'>{c['source_marker']}</span> <b>{c['filename']}</b> (Trang {c['page']})", unsafe_allow_html=True)


def _tab_quiz(selected_doc, page_filter):
    st.markdown("<h2 style='font-family:Space Grotesk; font-weight:700;'>📝 Trắc nghiệm tương tác</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Kiểm tra kiến thức đã học với bộ câu hỏi trắc nghiệm tự động sinh ra.</p>", unsafe_allow_html=True)

    count = st.slider("Số lượng câu hỏi trắc nghiệm cần tạo", min_value=3, max_value=15, value=5)

    if st.button("Tạo bài kiểm tra trắc nghiệm"):
        with st.spinner("Đang thiết lập bộ câu hỏi..."):
            filters = {}
            if page_filter:
                filters["page"] = page_filter

            payload = {
                "document": selected_doc,
                "count": count,
                "filters": filters if filters else None
            }
            res = _api("POST", "/quiz", json=payload)
            if res and res.get("items"):
                st.session_state["active_quiz"] = res
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False

    if "active_quiz" in st.session_state:
        quiz_data = st.session_state["active_quiz"]

        # Display questions
        for idx, item in enumerate(quiz_data["items"]):
            st.markdown(f"<div style='font-family:Space Grotesk; font-size:1.15rem; font-weight:600; margin-top:25px;'>Câu {idx+1}: {item['question']}</div>", unsafe_allow_html=True)

            # Make sure we track selection in state
            key = f"q_{idx}"
            options = item["options"]

            # Radio selection
            selected_option = st.radio(
                "Chọn đáp án đúng:",
                options,
                key=key,
                index=None,
                disabled=st.session_state["quiz_submitted"]
            )

            if selected_option:
                st.session_state["quiz_answers"][idx] = options.index(selected_option)

            # If submitted, show result for this question
            if st.session_state["quiz_submitted"]:
                user_ans = st.session_state["quiz_answers"].get(idx)
                correct_ans = item["correct_index"]

                if user_ans == correct_ans:
                    st.success("✅ Chính xác!")
                else:
                    st.error(f"❌ Chưa đúng! Đáp án đúng: {options[correct_ans]}")

                st.markdown(f"<div style='font-size:0.9rem; color:#94a3b8; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px;'><b>Giải thích:</b> {item['explanation']}</div>", unsafe_allow_html=True)

        # Submit Quiz Button
        st.markdown("<br>", unsafe_allow_html=True)
        if not st.session_state["quiz_submitted"]:
            if st.button("Nộp bài kiểm tra", use_container_width=True):
                st.session_state["quiz_submitted"] = True
                st.rerun()
        else:
            if st.button("Làm bài kiểm tra mới", use_container_width=True):
                del st.session_state["active_quiz"]
                st.session_state["quiz_submitted"] = False
                st.rerun()


def _tab_flashcards(selected_doc, page_filter):
    st.markdown("<h2 style='font-family:Space Grotesk; font-weight:700;'>🗂️ Thẻ ghi nhớ (Flashcards)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Học nhanh nhớ lâu các thuật ngữ và định nghĩa chính.</p>", unsafe_allow_html=True)

    count = st.slider("Số lượng thẻ ghi nhớ cần tạo", min_value=5, max_value=20, value=8)

    if st.button("Tạo bộ thẻ Flashcards"):
        with st.spinner("Đang thiết kế bộ thẻ ôn tập..."):
            filters = {}
            if page_filter:
                filters["page"] = page_filter

            payload = {
                "document": selected_doc,
                "count": count,
                "filters": filters if filters else None
            }
            res = _api("POST", "/flashcards", json=payload)
            if res and res.get("cards"):
                st.session_state["active_flashcards"] = res
                st.session_state["flashcard_index"] = 0
                st.session_state["flashcard_flipped"] = False

    if "active_flashcards" in st.session_state:
        fc_data = st.session_state["active_flashcards"]
        idx = st.session_state["flashcard_index"]
        card = fc_data["cards"][idx]

        # Navigation bar
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀️ Trước", disabled=(idx == 0), use_container_width=True):
                st.session_state["flashcard_index"] -= 1
                st.session_state["flashcard_flipped"] = False
                st.rerun()
        with col2:
            st.markdown(f"<div style='text-align:center; font-weight:600; color:#94a3b8;'>Thẻ {idx+1} / {len(fc_data['cards'])}</div>", unsafe_allow_html=True)
        with col3:
            if st.button("Sau ▶️", disabled=(idx == len(fc_data["cards"]) - 1), use_container_width=True):
                st.session_state["flashcard_index"] += 1
                st.session_state["flashcard_flipped"] = False
                st.rerun()

        # Interactive Flip Card
        st.markdown("<div class='flashcard-container'>", unsafe_allow_html=True)
        if card.get("topic"):
            st.markdown(f"<div class='flashcard-topic'>🏷️ {card['topic']}</div>", unsafe_allow_html=True)

        if not st.session_state["flashcard_flipped"]:
            st.markdown(f"<div class='flashcard-content'>{card['front']}</div>", unsafe_allow_html=True)
            if st.button("🔄 Lật mặt sau", use_container_width=True):
                st.session_state["flashcard_flipped"] = True
                st.rerun()
        else:
            st.markdown(f"<div class='flashcard-content' style='color:#a5b4fc;'>{card['back']}</div>", unsafe_allow_html=True)
            if card.get("hint"):
                st.markdown(f"<div class='flashcard-hint'>💡 Gợi ý: {card['hint']}</div>", unsafe_allow_html=True)
            if st.button("🔄 Lật mặt trước", use_container_width=True):
                st.session_state["flashcard_flipped"] = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def run():
    # Load custom styles first
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>📓 Simple NotebookLM</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Hệ thống RAG hỗ trợ học tập và nghiên cứu thông thái của riêng bạn — v2.0</div>", unsafe_allow_html=True)

    selected_doc, page_filter = _sidebar()

    tabs = st.tabs(["💬 Hỏi đáp", "📝 Tóm tắt", "🧠 Trắc nghiệm", "🗂️ Flashcards"])

    with tabs[0]:
        _tab_chat(selected_doc, page_filter)

    with tabs[1]:
        _tab_summary(selected_doc, page_filter)

    with tabs[2]:
        _tab_quiz(selected_doc, page_filter)

    with tabs[3]:
        _tab_flashcards(selected_doc, page_filter)

if __name__ == "__main__":
    run()
