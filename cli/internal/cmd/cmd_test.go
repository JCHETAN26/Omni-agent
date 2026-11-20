package cmd

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var fakeAWS = "AKIA" + "Q3ZK7MW2RT9PLX4N" // assembled so this file holds no key

func execute(t *testing.T, args ...string) (string, error) {
	t.Helper()
	var out bytes.Buffer
	root := NewRoot(&out, &out)
	root.SetArgs(args)
	err := root.Execute()
	return out.String(), err
}

func exitCode(err error) int {
	var e *ExitError
	if errors.As(err, &e) {
		return e.Code
	}
	if err != nil {
		return -1
	}
	return 0
}

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func run(t *testing.T, dir string, name string, args ...string) {
	t.Helper()
	c := exec.Command(name, args...)
	c.Dir = dir
	if out, err := c.CombinedOutput(); err != nil {
		t.Fatalf("%s %v: %v\n%s", name, args, err, out)
	}
}

func TestCheckExitCodesFollowFailOn(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "Dockerfile"), "FROM python:3.12-slim\nUSER 1000\n") // low only
	out, err := execute(t, "check", "--fast", dir)
	if exitCode(err) != 0 || !strings.Contains(out, "1 low") {
		t.Fatalf("low finding should pass at fail-on=high: code=%d\n%s", exitCode(err), out)
	}
	if _, err := execute(t, "check", "--fast", "--fail-on", "low", dir); exitCode(err) != 1 {
		t.Fatalf("fail-on=low should exit 1, got %v", err)
	}
	if _, err := execute(t, "check", "--fail-on", "severe", dir); exitCode(err) != -1 {
		t.Fatalf("bad --fail-on should be a usage error, got %v", err)
	}
	out, err = execute(t, "check", "--fast", "--json", dir)
	if err != nil || !strings.Contains(out, `"rule": "docker/no-healthcheck"`) {
		t.Fatalf("json output: %v\n%s", err, out)
	}
}

// The hook checks what is staged, not the working tree: a secret staged then
// fixed only in the working tree must still block the commit.
func TestHookBlocksStagedSecrets(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	repo := t.TempDir()
	run(t, repo, "git", "init", "-q")
	if out, err := execute(t, "hooks", "install", repo); err != nil || !strings.Contains(out, "pre-commit") {
		t.Fatalf("install: %v %s", err, out)
	}
	hook, err := os.ReadFile(filepath.Join(repo, ".git", "hooks", "pre-commit"))
	if err != nil || !strings.Contains(string(hook), "omni check --staged") {
		t.Fatalf("hook not written: %v", err)
	}
	// Reinstalling over our own hook is fine; over a foreign hook needs --force.
	if _, err := execute(t, "hooks", "install", repo); err != nil {
		t.Fatalf("reinstall: %v", err)
	}
	write(t, filepath.Join(repo, ".git", "hooks", "pre-commit"), "#!/bin/sh\necho custom\n")
	if _, err := execute(t, "hooks", "install", repo); err == nil {
		t.Fatal("overwrote a foreign hook without --force")
	}

	app := filepath.Join(repo, "app.py")
	write(t, app, "key = '"+fakeAWS+"'\n")
	run(t, repo, "git", "add", "app.py")
	write(t, app, "key = os.environ['KEY']\n") // fixed on disk, not staged
	out, err := execute(t, "check", "--staged", repo)
	if exitCode(err) != 1 || !strings.Contains(out, "secret/aws-access-key") {
		t.Fatalf("staged secret not blocked: code=%d\n%s", exitCode(err), out)
	}
	run(t, repo, "git", "add", "app.py")
	if out, err := execute(t, "check", "--staged", repo); exitCode(err) != 0 {
		t.Fatalf("clean stage blocked: %v\n%s", err, out)
	}
}

func TestNewServiceCommand(t *testing.T) {
	dir := t.TempDir()
	out, err := execute(t, "new", "service", "notes-api", "--dir", dir, "--no-check")
	if err != nil || !strings.Contains(out, "deploy/k8s.yaml") {
		t.Fatalf("%v\n%s", err, out)
	}
	if _, err := execute(t, "new", "service", "notes-api", "--dir", dir, "--no-check"); err == nil {
		t.Fatal("second generate should refuse to overwrite")
	}
}
