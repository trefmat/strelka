function getQuoteSize() {
  const input = document.getElementById('quoteSize');
  const raw = Number.parseInt(input?.value ?? '420', 10);
  if (Number.isNaN(raw)) return 420;
  return Math.max(120, Math.min(2400, raw));
}

function toSafeInt(value, fallback = 0) {
  const n = Number.parseInt(String(value), 10);
  return Number.isNaN(n) ? fallback : n;
}

function openReader(book, start = 0, end = 0) {
  if (!book) return;
  const params = new URLSearchParams({
    book,
    start: String(Math.max(0, toSafeInt(start, 0))),
    end: String(Math.max(0, toSafeInt(end, 0))),
  });
  window.open(`/reader?${params.toString()}`, '_blank');
}

function pickFocusStart(item) {
  return item.focus_start ?? item.offset_start ?? 0;
}

function pickFocusEnd(item) {
  return item.focus_end ?? item.offset_end ?? pickFocusStart(item);
}

const uiState = {
  lastUploadedBook: '',
};

function setLastUploadedBook(book) {
  uiState.lastUploadedBook = book || '';
  const btn = document.getElementById('readBookBtn');
  if (!btn) return;
  if (uiState.lastUploadedBook) {
    btn.classList.remove('hidden');
  } else {
    btn.classList.add('hidden');
  }
}

function bindOpenButtons(root) {
  const container = root || document;
  container.querySelectorAll('.openBookBtn').forEach((btn) => {
    btn.addEventListener('click', () => {
      openReader(btn.dataset.book, btn.dataset.start, btn.dataset.end);
    });
  });
}

async function loadPreloadedMenu() {
  const select = document.getElementById('preloadedBook');
  if (!select) return;

  const previous = select.value;
  select.innerHTML = '<option value="">Выберите предзагруженную книгу</option>';

  try {
    const res = await fetch('/books/preloaded');
    const data = await res.json();
    if (!res.ok) {
      return;
    }
    const books = Array.isArray(data.books) ? data.books : [];
    for (const item of books) {
      if (!item || !item.book) continue;
      const opt = document.createElement('option');
      opt.value = item.book;
      const kb = item.size_bytes ? `${Math.round(item.size_bytes / 1024)} KB` : 'txt';
      opt.textContent = `${item.book} (${kb})`;
      select.appendChild(opt);
    }
    if (previous) {
      select.value = previous;
    }
  } catch (err) {
  }
}

async function loadPreloadedBook() {
  const out = document.getElementById('uploadResult');
  const select = document.getElementById('preloadedBook');
  const book = select?.value?.trim();
  if (!book) {
    out.textContent = 'Выберите книгу из предустановленных';
    return;
  }

  out.textContent = 'Обработка...';
  const res = await fetch('/books/load_preloaded', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ book }),
  });
  const data = await res.json();
  out.textContent = res.ok
    ? `OK: ${data.book}, chunks=${data.chunks_added}`
    : `Ошибка: ${data.detail ?? 'unknown'}`;
  if (res.ok && data.book) {
    setLastUploadedBook(data.book);
  }
}

async function uploadBook() {
  const out = document.getElementById('uploadResult');
  const fileInput = document.getElementById('bookFile');
  if (!fileInput.files.length) {
    out.textContent = 'Выберите файл .txt, .fb2 или .epub';
    return;
  }

  out.textContent = 'Обработка...';
  const form = new FormData();
  form.append('file', fileInput.files[0]);
  const res = await fetch('/books/upload', { method: 'POST', body: form });
  const data = await res.json();
  out.textContent = res.ok ? `OK: ${data.book}, chunks=${data.chunks_added}` : `Ошибка: ${data.detail ?? 'unknown'}`;
  if (res.ok && data.book) {
    setLastUploadedBook(data.book);
  }
}

const searchState = {
  query: '',
  page: 1,
  pageSize: 5,
};

function renderSearchSnippets(data) {
  const out = document.getElementById('searchResult');
  if (!data.snippets.length) {
    out.textContent = 'Ничего не найдено';
    return;
  }

  const quotes = data.snippets.map((s, i) => {
    const rank = s.rank ?? ((data.page - 1) * data.page_size + i + 1);
    return `
      <div class="quote">
        <b>${rank}. ${s.book}</b> [${s.offset_start}-${s.offset_end}] score=${s.score}<br/>
        ${s.quote}
        <div class="quote-actions">
          <button class="btn-compact openBookBtn" data-book="${s.book}" data-start="${pickFocusStart(s)}" data-end="${pickFocusEnd(s)}">Открыть в книге</button>
        </div>
      </div>
    `;
  }).join('');

  const pager = `
    <div class="pager">
      <button id="searchPrevBtn" ${data.has_prev ? '' : 'disabled'}>Назад</button>
      <span>Страница ${data.page} из ${data.total_pages}, всего ${data.total}</span>
      <button id="searchNextBtn" ${data.has_next ? '' : 'disabled'}>Вперед</button>
    </div>
  `;

  out.innerHTML = `${quotes}${pager}`;

  const prevBtn = document.getElementById('searchPrevBtn');
  const nextBtn = document.getElementById('searchNextBtn');
  if (prevBtn) {
    prevBtn.addEventListener('click', () => fetchSearchPage(searchState.page - 1));
  }
  if (nextBtn) {
    nextBtn.addEventListener('click', () => fetchSearchPage(searchState.page + 1));
  }
  bindOpenButtons(out);
}

async function fetchSearchPage(page) {
  const out = document.getElementById('searchResult');
  out.textContent = 'Обработка...';
  const res = await fetch('/search/snippets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: searchState.query,
      page,
      page_size: searchState.pageSize,
      quote_size: getQuoteSize(),
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    out.textContent = `Ошибка: ${JSON.stringify(data)}`;
    return;
  }
  searchState.page = data.page ?? page;
  renderSearchSnippets(data);
}

async function searchSnippets() {
  const out = document.getElementById('searchResult');
  const query = document.getElementById('searchQuery').value.trim();
  if (!query) {
    out.textContent = 'Введите запрос';
    return;
  }
  searchState.query = query;
  searchState.page = 1;
  await fetchSearchPage(1);
}

async function askQuestion() {
  const out = document.getElementById('askResult');
  const question = document.getElementById('askQuery').value.trim();
  if (!question) {
    out.textContent = 'Введите вопрос';
    return;
  }

  out.textContent = 'Обработка...';
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: 5, quote_size: getQuoteSize() }),
  });
  const data = await res.json();
  if (!res.ok) {
    out.textContent = `Ошибка: ${JSON.stringify(data)}`;
    return;
  }
  const suggestions = Array.isArray(data.suggestions) ? data.suggestions : [];
  if (data.message === 'clarify_needed') {
    const tips = suggestions.length
      ? `<br/><br/><b>Попробуйте уточнить так:</b><ul class="suggestions">${suggestions.map((q) => `<li>${q}</li>`).join('')}</ul>`
      : '';
    out.innerHTML = `<b>Ответ:</b> ${data.answer}${tips}`;
    return;
  }
  const sources = (data.sources || []).map((s, i) => `
    <div class="quote">
      <b>${i + 1}. ${s.book}</b> [${s.offset_start}-${s.offset_end}] score=${s.score}<br/>
      ${s.quote}
      <div class="quote-actions">
        <button class="btn-compact openBookBtn" data-book="${s.book}" data-start="${pickFocusStart(s)}" data-end="${pickFocusEnd(s)}">Открыть в книге</button>
      </div>
    </div>
  `).join('');
  const tips = suggestions.length
    ? `<br/><br/><b>Как переформулировать вопрос:</b><ul class="suggestions">${suggestions.map((q) => `<li>${q}</li>`).join('')}</ul>`
    : '';
  out.innerHTML = `<b>Ответ:</b> ${data.answer}<br/><br/><b>Источники:</b>${sources || '<br/>нет'}${tips}`;
  bindOpenButtons(out);
}

document.getElementById('uploadBtn').addEventListener('click', uploadBook);
document.getElementById('loadPreloadedBtn').addEventListener('click', loadPreloadedBook);
document.getElementById('searchBtn').addEventListener('click', searchSnippets);
document.getElementById('askBtn').addEventListener('click', askQuestion);

document.getElementById('readBookBtn').addEventListener('click', () => {
  if (!uiState.lastUploadedBook) return;
  openReader(uiState.lastUploadedBook, 0, 0);
});

loadPreloadedMenu();
