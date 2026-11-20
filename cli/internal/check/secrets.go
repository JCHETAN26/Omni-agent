package check

import (
	"math"
	"path"
	"regexp"
	"strings"
)

type secretPattern struct {
	rule string
	re   *regexp.Regexp
	what string
}

var secretPatterns = []secretPattern{
	{"secret/private-key", regexp.MustCompile(`-----BEGIN ((RSA|EC|DSA|OPENSSH|ENCRYPTED) )?PRIVATE KEY-----`), "private key"},
	{"secret/aws-access-key", regexp.MustCompile(`\b(AKIA|ASIA)[0-9A-Z]{16}\b`), "AWS access key id"},
	{"secret/github-token", regexp.MustCompile(`\b(gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b`), "GitHub token"},
	{"secret/anthropic-key", regexp.MustCompile(`\bsk-ant-[A-Za-z0-9_-]{20,}`), "Anthropic API key"},
	{"secret/openai-key", regexp.MustCompile(`\bsk-(proj-|svcacct-)?[A-Za-z0-9_-]{32,}`), "OpenAI API key"},
	{"secret/slack-token", regexp.MustCompile(`\bxox[abposr]-[A-Za-z0-9-]{10,}`), "Slack token"},
	{"secret/google-api-key", regexp.MustCompile(`\bAIza[0-9A-Za-z_-]{35}\b`), "Google API key"},
}

// key = "value" style assignments of credential-named settings.
var assignment = regexp.MustCompile(
	`(?i)\b([a-z0-9_.-]*(api[_-]?key|secret|passw(or)?d|token|access[_-]?key|credentials?))["']?\s*[:=]\s*["']([^"'\s]{12,})["']`)

var placeholders = []string{"example", "changeme", "change-me", "placeholder", "dummy", "your", "xxxx", "<", "${", "{{", "redacted", "test"}

const allowMarker = "omni:allow-secret"

func entropy(s string) float64 {
	counts := map[rune]int{}
	for _, r := range s {
		counts[r]++
	}
	var h float64
	n := float64(len([]rune(s)))
	for _, c := range counts {
		p := float64(c) / n
		h -= p * math.Log2(p)
	}
	return h
}

func looksPlaceholder(v string) bool {
	low := strings.ToLower(v)
	for _, p := range placeholders {
		if strings.Contains(low, p) {
			return true
		}
	}
	return false
}

// isEnvFile matches .env and .env.<stage>, but not the committed templates.
func isEnvFile(rel string) bool {
	base := path.Base(rel)
	if base != ".env" && !strings.HasPrefix(base, ".env.") {
		return false
	}
	switch strings.TrimPrefix(base, ".env.") {
	case "example", "sample", "template", "dist":
		return false
	}
	return true
}

func scanSecrets(f file) []Finding {
	var out []Finding
	lines := strings.Split(string(f.data), "\n")
	if isEnvFile(f.rel) {
		for i, line := range lines {
			k, v, ok := strings.Cut(strings.TrimSpace(line), "=")
			if ok && !strings.HasPrefix(k, "#") && strings.Trim(v, `"' `) != "" {
				out = append(out, Finding{
					Rule: "secret/env-file", Severity: High, File: f.rel, Line: i + 1,
					Message: "environment file with values is in the tree",
					Fix:     "add it to .gitignore and commit a .env.example with empty values instead",
				})
				break
			}
		}
	}
	for i, line := range lines {
		if strings.Contains(line, allowMarker) {
			continue
		}
		matched := false
		for _, p := range secretPatterns {
			if m := p.re.FindString(line); m != "" && !looksPlaceholder(m) {
				out = append(out, Finding{
					Rule: p.rule, Severity: Critical, File: f.rel, Line: i + 1,
					Message: p.what + " in source: " + redact(m),
					Fix:     "revoke the credential, remove it from history, and load it from the environment or a secret store",
				})
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		if m := assignment.FindStringSubmatch(line); m != nil {
			value := m[4]
			if !looksPlaceholder(value) && entropy(value) >= 3.5 {
				out = append(out, Finding{
					Rule: "secret/hardcoded-credential", Severity: High, File: f.rel, Line: i + 1,
					Message: "hard-coded value for " + m[1] + ": " + redact(value),
					Fix:     "read it from an environment variable or a mounted secret",
				})
			}
		}
	}
	return out
}

func redact(s string) string {
	if len(s) <= 8 {
		return "****"
	}
	return s[:4] + strings.Repeat("*", 8) + s[len(s)-2:]
}
