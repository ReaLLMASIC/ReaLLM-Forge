export type TargetMode = "include" | "exclude" | "duty";
export type TargetRule = {
  id: number;
  token: number;
  start: number;
  mode: TargetMode;
  period: number;
  percent: number;
};
export type TargetRuleDraft = Omit<TargetRule, "id">;

export function dutyOnSteps(period: number, percent: number) {
  return Math.round(period * percent / 100);
}

export function validateTargetRule(rule: TargetRuleDraft, targetCount: number, liveStep: number) {
  if (!Number.isInteger(rule.token) || rule.token < 0 || rule.token >= targetCount) throw new Error("Choose an original target token from this run.");
  if (!Number.isInteger(rule.start) || rule.start < liveStep || rule.start > 50_000) throw new Error(`Choose a start iteration from ${liveStep} to 50,000. Recorded history cannot be changed.`);
  if (!["include", "exclude", "duty"].includes(rule.mode)) throw new Error("Choose include, exclude, or duty cycle.");
  if (rule.mode === "duty") {
    if (!Number.isInteger(rule.period) || rule.period < 1 || rule.period > 50_000) throw new Error("Duty period must be 1–50,000 whole iterations.");
    if (!Number.isFinite(rule.percent) || rule.percent < 0 || rule.percent > 100) throw new Error("Included percentage must be from 0 to 100.");
  }
  return { ...rule };
}

export function activeTargetRule(rules: readonly TargetRule[], token: number, step: number) {
  let active: TargetRule | undefined;
  for (const rule of rules) {
    if (rule.token === token && rule.start <= step && (!active || rule.start > active.start || (rule.start === active.start && rule.id > active.id))) active = rule;
  }
  return active;
}

export function includedAt(rule: TargetRule | undefined, step: number) {
  if (!rule || rule.mode === "include") return true;
  if (rule.mode === "exclude") return false;
  return (step - rule.start) % rule.period < dutyOnSteps(rule.period, rule.percent);
}

export function includedTargets(rules: readonly TargetRule[], targetCount: number, step: number) {
  return Array.from({ length: targetCount }, (_, token) => token).filter((token) => includedAt(activeTargetRule(rules, token, step), step));
}

export function describeTargetRule(rule: TargetRuleDraft) {
  if (rule.mode === "include") return "include";
  if (rule.mode === "exclude") return "exclude";
  return `${rule.percent}% duty · ${dutyOnSteps(rule.period, rule.percent)}/${rule.period} on`;
}
