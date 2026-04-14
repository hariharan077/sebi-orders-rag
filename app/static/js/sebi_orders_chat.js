(function () {
    const shell = document.querySelector("[data-sebi-chat]");
    if (!shell) {
        return;
    }

    const thread = shell.querySelector("[data-chat-thread]");
    const historyList = shell.querySelector("[data-chat-history]");
    const form = shell.querySelector("[data-chat-form]");
    const input = shell.querySelector("[data-chat-input]");
    const sendButton = shell.querySelector("[data-chat-send]");
    const newChatButton = shell.querySelector("[data-new-chat]");
    const queryEndpoint = shell.dataset.queryEndpoint;
    const sessionsEndpoint = shell.dataset.sessionsEndpoint;
    const sessionHistoryEndpointTemplate = shell.dataset.sessionHistoryEndpointTemplate;
    let currentSessionId = null;
    let isPending = false;
    let sessionSummaries = [];

    function resizeInput() {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 220) + "px";
    }

    function scrollToLatest() {
        thread.scrollTop = thread.scrollHeight;
    }

    function setPending(nextPending) {
        isPending = nextPending;
        input.disabled = nextPending;
        sendButton.disabled = nextPending;
        newChatButton.disabled = nextPending;
        historyList.querySelectorAll("button").forEach(function (button) {
            button.disabled = nextPending;
        });
        if (!nextPending) {
            input.focus();
            resizeInput();
        }
    }

    function clearThread() {
        thread.innerHTML = "";
    }

    function setCurrentSessionId(nextSessionId) {
        currentSessionId = nextSessionId || null;
        renderSessionList();
    }

    function pageLabel(citation) {
        if (
            !citation ||
            typeof citation.page_start !== "number" ||
            typeof citation.page_end !== "number"
        ) {
            return "";
        }
        if (citation.page_start === citation.page_end) {
            return "p. " + citation.page_start;
        }
        return "pp. " + citation.page_start + "-" + citation.page_end;
    }

    function stripInlineCitations(text) {
        const rawText = typeof text === "string" ? text : "";
        return rawText
            .replace(/\[(\d+)\]/g, "")
            .replace(/[ \t]+([,.;:!?])/g, "$1")
            .replace(/\(\s+/g, "(")
            .replace(/\s+\)/g, ")")
            .replace(/[ \t]{2,}/g, " ")
            .trim();
    }

    function resolveTitleUrl(citation) {
        if (!citation) {
            return null;
        }
        return citation.title_url || citation.detail_url || citation.pdf_url || citation.source_url || null;
    }

    function resolvePageUrl(citation) {
        if (!citation) {
            return null;
        }
        return citation.page_url || null;
    }

    function formatSourceMeta(citation) {
        const parts = [];
        const pages = pageLabel(citation);
        if (pages) {
            parts.push(pages);
        }
        if (citation && citation.domain && (citation.source_type === "official_web" || citation.source_type === "general_web")) {
            parts.push(citation.domain);
        } else if (citation && citation.record_key) {
            parts.push(citation.record_key);
        }
        return parts.join(" • ");
    }

    function appendParagraphs(container, text) {
        const sanitizedText = stripInlineCitations(text);
        const normalizedText = typeof sanitizedText === "string" && sanitizedText.trim() ? sanitizedText : "";
        const blocks = normalizedText.split(/\n{2,}/).filter(Boolean);
        const paragraphs = blocks.length > 0 ? blocks : [normalizedText || "No response returned."];
        paragraphs.forEach(function (block) {
            const paragraph = document.createElement("p");
            paragraph.textContent = block.trim();
            container.appendChild(paragraph);
        });
    }

    function appendMessage(role, text, options) {
        const safeOptions = options || {};
        const citations = Array.isArray(safeOptions.citations) ? safeOptions.citations : [];

        const message = document.createElement("article");
        message.className = "chat-message chat-message--" + role;

        const bubble = document.createElement("div");
        bubble.className = "chat-bubble";
        appendParagraphs(bubble, text);
        message.appendChild(bubble);

        if (role === "assistant" && citations.length > 0) {
            const sources = document.createElement("section");
            sources.className = "chat-sources";

            const heading = document.createElement("div");
            heading.className = "chat-sources-title";
            heading.textContent = "Sources";
            sources.appendChild(heading);

            const list = document.createElement("ul");
            list.className = "chat-sources-list";
            citations.forEach(function (citation) {
                const item = document.createElement("li");
                item.className = "chat-sources-item";

                const titleUrl = resolveTitleUrl(citation);
                const title = document.createElement(titleUrl ? "a" : "span");
                title.className = "chat-sources-item-title";
                title.textContent = citation.title;
                if (titleUrl) {
                    title.href = titleUrl;
                    title.target = "_blank";
                    title.rel = "noopener noreferrer";
                }

                const detail = document.createElement("div");
                detail.className = "chat-sources-item-detail";
                const pageUrl = resolvePageUrl(citation);
                const sourceMeta = formatSourceMeta(citation);
                if (pageUrl && pageLabel(citation)) {
                    const pageLink = document.createElement("a");
                    pageLink.className = "chat-sources-page-link";
                    pageLink.textContent = pageLabel(citation);
                    pageLink.href = pageUrl;
                    pageLink.target = "_blank";
                    pageLink.rel = "noopener noreferrer";
                    detail.appendChild(pageLink);

                    if (citation.record_key) {
                        const recordKey = document.createElement("span");
                        recordKey.className = "chat-sources-record-key";
                        recordKey.textContent = citation.record_key;
                        detail.appendChild(recordKey);
                    }
                } else {
                    detail.textContent = sourceMeta;
                }

                item.appendChild(title);
                item.appendChild(detail);
                list.appendChild(item);
            });
            sources.appendChild(list);
            message.appendChild(sources);
        }

        thread.appendChild(message);
        scrollToLatest();
        return message;
    }

    function renderTurns(turns) {
        clearThread();
        if (!Array.isArray(turns) || turns.length === 0) {
            return;
        }

        turns.forEach(function (turn) {
            appendMessage("user", turn.user_message || "", {});
            appendMessage("assistant", turn.assistant_message || "", {
                citations: Array.isArray(turn.citations) ? turn.citations : [],
            });
        });
    }

    function appendLoadingMessage() {
        const loading = document.createElement("article");
        loading.className = "chat-message chat-message--assistant chat-message--loading";

        const bubble = document.createElement("div");
        bubble.className = "chat-bubble chat-bubble--loading";

        const dots = document.createElement("div");
        dots.className = "chat-loading";
        dots.innerHTML = "<span></span><span></span><span></span>";

        bubble.appendChild(dots);
        loading.appendChild(bubble);
        thread.appendChild(loading);
        scrollToLatest();
        return loading;
    }

    function formatSessionTimestamp(value) {
        if (!value) {
            return "";
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
            return "";
        }
        return new Intl.DateTimeFormat(undefined, {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
        }).format(date);
    }

    function formatSessionMeta(summary) {
        return formatSessionTimestamp(summary.last_message_at || summary.updated_at || summary.created_at);
    }

    function renderSessionList() {
        historyList.innerHTML = "";

        if (!Array.isArray(sessionSummaries) || sessionSummaries.length === 0) {
            const emptyState = document.createElement("p");
            emptyState.className = "sebi-chat-history-empty";
            emptyState.textContent = "No saved chats yet.";
            historyList.appendChild(emptyState);
            return;
        }

        const fragment = document.createDocumentFragment();
        sessionSummaries.forEach(function (summary) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "sebi-chat-history-item";
            button.disabled = isPending;
            button.dataset.sessionId = summary.session_id;
            button.setAttribute("aria-pressed", currentSessionId === summary.session_id ? "true" : "false");
            if (currentSessionId === summary.session_id) {
                button.classList.add("is-active");
            }

            const title = document.createElement("span");
            title.className = "sebi-chat-history-item-title";
            title.textContent = summary.title;

            const preview = document.createElement("span");
            preview.className = "sebi-chat-history-item-preview";
            preview.textContent = summary.preview_text;

            button.appendChild(title);
            button.appendChild(preview);
            const metaText = formatSessionMeta(summary);
            if (metaText) {
                const meta = document.createElement("span");
                meta.className = "sebi-chat-history-item-meta";
                meta.textContent = metaText;
                button.appendChild(meta);
            }
            button.addEventListener("click", function () {
                if (isPending || summary.session_id === currentSessionId) {
                    return;
                }
                handleSessionSelection(summary.session_id);
            });

            fragment.appendChild(button);
        });

        historyList.appendChild(fragment);
    }

    async function fetchJson(url, options) {
        const response = await fetch(url, options);
        if (!response.ok) {
            let detail = "I’m temporarily unable to answer that safely right now.";
            try {
                const errorPayload = await response.json();
                if (errorPayload && typeof errorPayload.detail === "string" && errorPayload.detail.trim()) {
                    detail = errorPayload.detail.trim();
                }
            } catch (error) {
                // Ignore non-JSON failures and keep the fallback message.
            }
            throw new Error(detail);
        }
        return response.json();
    }

    async function refreshSessionList() {
        const payload = await fetchJson(sessionsEndpoint);
        sessionSummaries = Array.isArray(payload.sessions) ? payload.sessions : [];
        renderSessionList();
    }

    function buildSessionHistoryUrl(sessionId) {
        return sessionHistoryEndpointTemplate.replace("__SESSION_ID__", sessionId);
    }

    async function loadSessionHistory(sessionId) {
        const payload = await fetchJson(buildSessionHistoryUrl(sessionId));
        const resolvedSessionId = payload && payload.session ? payload.session.session_id : sessionId;
        setCurrentSessionId(resolvedSessionId);
        renderTurns(Array.isArray(payload.turns) ? payload.turns : []);
    }

    async function handleSessionSelection(sessionId) {
        if (!sessionId || isPending) {
            return;
        }

        setPending(true);
        try {
            await loadSessionHistory(sessionId);
        } catch (error) {
            const detail = error instanceof Error ? error.message : "Unable to load the selected chat.";
            clearThread();
            appendMessage("assistant", detail, {});
        } finally {
            setPending(false);
        }
    }

    function startNewChat() {
        if (isPending) {
            return;
        }
        setCurrentSessionId(null);
        clearThread();
        input.focus();
    }

    async function submitMessage() {
        if (isPending) {
            return;
        }

        const message = input.value.trim();
        if (!message) {
            return;
        }

        appendMessage("user", message, {});
        input.value = "";
        resizeInput();
        setPending(true);

        const loadingMessage = appendLoadingMessage();

        try {
            const payload = await fetchJson(queryEndpoint, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: message,
                    session_id: currentSessionId,
                }),
            });

            loadingMessage.remove();
            setCurrentSessionId(payload.session_id);
            appendMessage("assistant", payload.answer_text, {
                citations: Array.isArray(payload.citations) ? payload.citations : [],
            });
            try {
                await refreshSessionList();
                setCurrentSessionId(payload.session_id);
            } catch (error) {
                // Keep the active chat usable even if the sidebar refresh fails.
            }
        } catch (error) {
            loadingMessage.remove();
            const messageText = error instanceof Error ? error.message : "I’m temporarily unable to answer that safely right now.";
            appendMessage("assistant", messageText, {});
        } finally {
            setPending(false);
        }
    }

    async function initialize() {
        setPending(true);
        try {
            await refreshSessionList();
            setCurrentSessionId(null);
            clearThread();
        } catch (error) {
            const detail = error instanceof Error ? error.message : "Unable to load the saved chats.";
            setCurrentSessionId(null);
            clearThread();
            appendMessage("assistant", detail, {});
        } finally {
            setPending(false);
        }
    }

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        submitMessage();
    });

    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submitMessage();
        }
    });

    newChatButton.addEventListener("click", startNewChat);
    input.addEventListener("input", resizeInput);
    resizeInput();
    initialize();
}());
