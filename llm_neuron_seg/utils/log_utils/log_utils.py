import logging
def clear_existing_loggers():
    """Clear all existing loggers to prevent conflicts."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

def setup_logger(logfile_path):
    """Setup logger to always print time and level."""
    clear_existing_loggers()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler()
        ]
    )

def log_w_indent(text, indent, flag, symbol='>>'):
    """Log and add indent."""
    ind2col = {i: f"\x1b[{a}m" for i, a in enumerate([
        '1', '31', '33', '34', '35'])}
    reset = "\x1b[0m"

    if indent < 0 or indent >= len(ind2col):
        raise ValueError("Invalid indent level")

    if indent > 0:
        if flag == 'Question':
            prefix = '  Q:'
        elif flag == 'Response':
            prefix = '  LLM-A:'
        elif flag == 'Code':
            prefix = '  Code script test:'
        else:
            raise ValueError("Invalid flag")

        prefix += ind2col[indent] + (indent * 2) * symbol + ' ' + text + reset
    else:
        prefix = ind2col[indent] + text + reset

    logging.info(prefix)


if __name__ == "__main__":
    setup_logger(r"D:\GXQ\human_seg\neuron_seg_human\llm_neuron_seg\utils\log_utils\log.log")
    log_w_indent("This is a test question", 1, "Question")
    log_w_indent("This is a test response", 1, "Response")
