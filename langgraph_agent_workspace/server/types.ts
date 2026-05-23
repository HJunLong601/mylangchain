// 这个文件放“前后端都会关心”的核心类型。
// 当前没有做 monorepo 类型共享，为了学习清晰，先把后端主类型集中在这里。

export type AgentIntent = "chat" | "save_note" | "search_notes";

export type AgentMessageRole = "user" | "assistant";

export type AgentMessage = {
  role: AgentMessageRole;
  content: string;
};

export type DebugEvent = {
  node: string;
  message: string;
  data?: unknown;
};

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
