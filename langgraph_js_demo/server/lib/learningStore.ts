import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// 这个文件是“学习沉淀”的最小持久化层。
//
// 当前阶段先不用数据库，只用本地 JSON 文件：
// - 优点：简单、可读、适合学习
// - 局限：不适合多人并发写入，也不适合生产
//
// 后面如果要升级，可以把这里替换成 SQLite、Postgres 或其他存储。

export type LearningNoteKind = "concept" | "observation" | "question" | "todo";

export type LearningNote = {
  id: string;
  title: string;
  content: string;
  kind: LearningNoteKind;
  tags: string[];
  createdAt: string;
  updatedAt: string;
};

export type CreateLearningNoteInput = {
  title: string;
  content: string;
  kind?: LearningNoteKind;
  tags?: string[];
};

const currentDir = dirname(fileURLToPath(import.meta.url));
const dataFilePath = join(currentDir, "..", "data", "learning-notes.json");

function createId() {
  return `note_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function normalizeTags(tags: string[] | undefined) {
  return Array.from(
    new Set(
      (tags ?? [])
        .map((tag) => tag.trim())
        .filter(Boolean),
    ),
  );
}

async function ensureDataFile() {
  await mkdir(dirname(dataFilePath), {
    recursive: true,
  });

  try {
    await readFile(dataFilePath, "utf-8");
  } catch {
    await writeFile(dataFilePath, "[]", "utf-8");
  }
}

export async function listLearningNotes(): Promise<LearningNote[]> {
  await ensureDataFile();

  const rawContent = await readFile(dataFilePath, "utf-8");
  const notes = JSON.parse(rawContent) as LearningNote[];

  // 最新笔记排前面，方便在页面上看到刚刚沉淀的内容。
  return notes.sort((left, right) => (
    right.createdAt.localeCompare(left.createdAt)
  ));
}

export async function createLearningNote(
  input: CreateLearningNoteInput,
): Promise<LearningNote> {
  const title = input.title.trim();
  const content = input.content.trim();

  if (!title) {
    throw new Error("title is required");
  }
  if (!content) {
    throw new Error("content is required");
  }

  const now = new Date().toISOString();
  const note: LearningNote = {
    id: createId(),
    title,
    content,
    kind: input.kind ?? "observation",
    tags: normalizeTags(input.tags),
    createdAt: now,
    updatedAt: now,
  };

  const notes = await listLearningNotes();
  notes.unshift(note);
  await writeFile(dataFilePath, JSON.stringify(notes, null, 2), "utf-8");

  return note;
}
