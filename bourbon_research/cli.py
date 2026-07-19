import argparse
import json
import sys

from dotenv import load_dotenv

from bourbon_research.service import ResearchDesk


def _print(value, as_json=False):
    if as_json:
        print(json.dumps(value, indent=2, default=str))
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"{key}: {item}")
    elif isinstance(value, list):
        for item in value:
            print(item if isinstance(item, str) else json.dumps(item, default=str))
    else:
        print(value)


def build_parser():
    parser = argparse.ArgumentParser(prog="bourbon-research", description="Persistent evidence-first research desk")
    parser.add_argument("--workspace", default=".bourbon-research")
    parser.add_argument("--project", help="Project slug; defaults to the current project")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    project = commands.add_parser("project")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    create = project_commands.add_parser("create")
    create.add_argument("subject")
    create.add_argument("--objective")
    project_commands.add_parser("list")
    use = project_commands.add_parser("use")
    use.add_argument("slug")

    research = commands.add_parser("research")
    research_commands = research.add_subparsers(dest="research_command", required=True)
    research_commands.add_parser("plan")
    run = research_commands.add_parser("run")
    run.add_argument("--max-sources", type=int, default=10)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--source-policy",
        choices=["authoritative", "exclude-community", "all"],
        default="exclude-community",
    )
    research_commands.add_parser("consolidate")

    commands.add_parser("status")
    commands.add_parser("sources")
    commands.add_parser("claims")
    commands.add_parser("contradictions")
    memory = commands.add_parser("memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    trace = memory_commands.add_parser("trace")
    trace.add_argument("--token-budget", type=int, default=1000)
    trace.add_argument("--query", help="Current task or question used to rank memory")
    report = commands.add_parser("report")
    report.add_argument("--output")
    session = commands.add_parser("session")
    session_commands = session.add_subparsers(dest="session_command", required=True)
    session_commands.add_parser("changes")
    return parser


def main(argv=None):
    # Load the nearest .env without replacing values explicitly supplied by the
    # parent process. This happens before ResearchDesk selects its providers.
    load_dotenv(override=False)
    args = build_parser().parse_args(argv)
    desk = ResearchDesk(workspace=args.workspace)
    try:
        if args.command == "project":
            if args.project_command == "create":
                value = desk.create_project(args.subject, args.objective).__dict__
            elif args.project_command == "list":
                value = desk.repository.list_projects()
            else:
                desk.repository.get_project(args.slug)
                desk.repository.set_current_project(args.slug)
                value = {"current_project": args.slug}
        elif args.command == "research":
            if args.research_command == "plan":
                value = desk.plan(args.project)
            elif args.research_command == "run":
                value = desk.run(
                    args.project,
                    max_sources=args.max_sources,
                    dry_run=args.dry_run,
                    source_policy=args.source_policy,
                )
            else:
                value = {"derived_memory": desk.consolidate(args.project)}
        elif args.command == "status":
            value = desk.status(args.project)
        elif args.command == "sources":
            value = desk.repository.sources(desk.project(args.project).id)
        elif args.command == "claims":
            value = desk.repository.claims(desk.project(args.project).id)
        elif args.command == "contradictions":
            value = desk.repository.contradictions(desk.project(args.project).id)
        elif args.command == "memory":
            value = desk.memory_trace(args.project, args.token_budget, args.query)
        elif args.command == "report":
            value = {"report": str(desk.report(args.project, args.output))}
        elif args.command == "session":
            value = desk.repository.session_changes(desk.project(args.project).id)
        _print(value, args.json)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        desk.close()


if __name__ == "__main__":
    raise SystemExit(main())
