async function uploadBook() {
  const out = document.getElementById('uploadResult');
  const fileInput = document.getElementById('bookFile');
  if (!fileInput.files.length) {
    out.textContent = 'Выберите .txt файл';
    return;
  }
  const form = new FormData();
  form.append('file', fileInput.files[0]);
  const res = await fetch('/books/upload', { method: 'POST', body: form });
  const data = await res.json();
  out.textContent = res.ok ? `OK: ${data.book}, chunks=${data.chunks_added}` : `Ошибка: ${data.detail ?? 'unknown'}`;
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
}

async function fetchSearchPage(page) {
  const out = document.getElementById('searchResult');
  const res = await fetch('/search/snippets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: searchState.query, page, page_size: searchState.pageSize }),
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
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: 5 }),
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
    </div>
  `).join('');
  const tips = suggestions.length
    ? `<br/><br/><b>Как переформулировать вопрос:</b><ul class="suggestions">${suggestions.map((q) => `<li>${q}</li>`).join('')}</ul>`
    : '';
  out.innerHTML = `<b>Ответ:</b> ${data.answer}<br/><br/><b>Источники:</b>${sources || '<br/>нет'}${tips}`;
}

document.getElementById('uploadBtn').addEventListener('click', uploadBook);
document.getElementById('searchBtn').addEventListener('click', searchSnippets);
document.getElementById('askBtn').addEventListener('click', askQuestion);
