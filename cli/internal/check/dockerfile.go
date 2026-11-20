package check

import (
	"path"
	"regexp"
	"strings"
)

func isDockerfile(rel string) bool {
	base := strings.ToLower(path.Base(rel))
	return base == "dockerfile" || strings.HasPrefix(base, "dockerfile.") || strings.HasSuffix(base, ".dockerfile")
}

type instruction struct {
	line int
	cmd  string // upper-case keyword
	args string
}

// parseDockerfile joins backslash continuations and drops comments.
func parseDockerfile(data []byte) []instruction {
	var out []instruction
	var buf strings.Builder
	start := 0
	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if buf.Len() == 0 {
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			start = i + 1
		} else if strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasSuffix(line, "\\") {
			buf.WriteString(strings.TrimSuffix(line, "\\"))
			buf.WriteString(" ")
			continue
		}
		buf.WriteString(line)
		cmd, args, _ := strings.Cut(strings.TrimSpace(buf.String()), " ")
		out = append(out, instruction{line: start, cmd: strings.ToUpper(cmd), args: strings.TrimSpace(args)})
		buf.Reset()
	}
	return out
}

var (
	curlPipe  = regexp.MustCompile(`\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b`)
	sudoUse   = regexp.MustCompile(`(^|[;&|]\s*)sudo\b`)
	secretKey = regexp.MustCompile(`(?i)(api[_-]?key|secret|passw(or)?d|token|access[_-]?key)`)
)

type stage struct {
	image, alias string
	line         int
	user         string
	userLine     int
	hasHealth    bool
}

func scanDockerfile(f file) []Finding {
	var out []Finding
	add := func(rule string, sev Severity, line int, msg, fix string) {
		out = append(out, Finding{Rule: rule, Severity: sev, File: f.rel, Line: line, Message: msg, Fix: fix})
	}
	var stages []*stage
	aliases := map[string]bool{}
	for _, in := range parseDockerfile(f.data) {
		switch in.cmd {
		case "FROM":
			fields := strings.Fields(in.args)
			var image string
			for _, fl := range fields {
				if !strings.HasPrefix(fl, "--") {
					image = fl
					break
				}
			}
			st := &stage{image: image, line: in.line}
			for i, fl := range fields {
				if strings.EqualFold(fl, "AS") && i+1 < len(fields) {
					st.alias = strings.ToLower(fields[i+1])
				}
			}
			stages = append(stages, st)
			if !aliases[strings.ToLower(image)] && image != "scratch" && !strings.Contains(image, "$") &&
				!strings.Contains(image, "@sha256:") {
				name := image[strings.LastIndex(image, "/")+1:]
				if !strings.Contains(name, ":") || strings.HasSuffix(image, ":latest") {
					add("docker/unpinned-base", Medium, in.line, "base image "+image+" is not pinned to a version",
						"pin a version tag, or a digest (image@sha256:...) for reproducible builds")
				}
			}
			if st.alias != "" {
				aliases[st.alias] = true
			}
		case "USER":
			if len(stages) > 0 {
				stages[len(stages)-1].user = strings.Fields(in.args + " ")[0]
				stages[len(stages)-1].userLine = in.line
			}
		case "HEALTHCHECK":
			if len(stages) > 0 && !strings.EqualFold(strings.TrimSpace(in.args), "NONE") {
				stages[len(stages)-1].hasHealth = true
			}
		case "ADD":
			if strings.Contains(in.args, "http://") || strings.Contains(in.args, "https://") {
				add("docker/add-remote", Medium, in.line, "ADD fetches a remote URL without checksum verification",
					"download with RUN curl and verify a checksum, or use ADD --checksum=sha256:...")
			}
		case "RUN":
			if curlPipe.MatchString(in.args) {
				add("docker/curl-pipe-shell", High, in.line, "RUN pipes a downloaded script straight into a shell",
					"download, verify a checksum or signature, then execute")
			}
			if sudoUse.MatchString(in.args) {
				add("docker/sudo", Medium, in.line, "RUN uses sudo inside the image", "run the step as root before switching USER")
			}
		case "ENV", "ARG":
			for _, kv := range strings.Fields(in.args) {
				k, v, ok := strings.Cut(kv, "=")
				if ok && secretKey.MatchString(k) && strings.Trim(v, `"'`) != "" && !strings.HasPrefix(v, "$") {
					add("docker/secret-in-image", High, in.line, in.cmd+" "+k+" bakes a credential into the image layers",
						"pass it at runtime (environment, mounted secret) or use RUN --mount=type=secret")
				}
			}
		}
	}
	if len(stages) == 0 {
		return out
	}
	final := stages[len(stages)-1]
	user := strings.ToLower(strings.SplitN(final.user, ":", 2)[0])
	nonrootImage := strings.Contains(final.image, "nonroot") || strings.Contains(final.image, "unprivileged") ||
		strings.Contains(final.image, "chainguard")
	if user == "root" || user == "0" {
		add("docker/root-user", High, final.userLine, "final stage runs as root",
			"create an unprivileged user and switch to it with USER")
	} else if user == "" && !nonrootImage {
		add("docker/root-user", High, final.line, "final stage never sets USER, so it runs as root",
			"create an unprivileged user and switch to it with USER")
	}
	if !final.hasHealth {
		add("docker/no-healthcheck", Low, final.line, "final stage has no HEALTHCHECK",
			"add a HEALTHCHECK (Kubernetes probes cover this when deployed there)")
	}
	return out
}
