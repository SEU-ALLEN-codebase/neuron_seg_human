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


def parse_json(function_args: str, finished: bool):
    """
    GPT may generate non-standard JSON format string, which contains '\n' in string value, leading to error when using
    `json.loads()`.
    Here we implement a parser to extract code directly from non-standard JSON string.
    :return: code string if successfully parsed otherwise None
    """
    parser_log = {
        'met_begin_{': False,
        'begin_"code"': False,
        'end_"code"': False,
        'met_:': False,
        'met_end_}': False,
        'met_end_code_"': False,
        "code_begin_index": 0,
        "code_end_index": 0
    }
    try:
        for index, char in enumerate(function_args):
            if char == '{':
                parser_log['met_begin_{'] = True
            elif parser_log['met_begin_{'] and char == '"':
                if parser_log['met_:']:
                    if finished:
                        parser_log['code_begin_index'] = index + 1
                        break
                    else:
                        if index + 1 == len(function_args):
                            return None
                        else:
                            temp_code_str = function_args[index + 1:]
                            if '\n' in temp_code_str:
                                try:
                                    return json.loads(function_args + '"}')['code']
                                except json.JSONDecodeError:
                                    try:
                                        return json.loads(function_args + '}')['code']
                                    except json.JSONDecodeError:
                                        try:
                                            return json.loads(function_args)['code']
                                        except json.JSONDecodeError:
                                            if temp_code_str[-1] in ('"', '\n'):
                                                return None
                                            else:
                                                return temp_code_str.strip('\n')
                            else:
                                return json.loads(function_args + '"}')['code']
                elif parser_log['begin_"code"']:
                    parser_log['end_"code"'] = True
                else:
                    parser_log['begin_"code"'] = True
            elif parser_log['end_"code"'] and char == ':':
                parser_log['met_:'] = True
            else:
                continue
        if finished:
            for index, char in enumerate(function_args[::-1]):
                back_index = -1 - index
                if char == '}':
                    parser_log['met_end_}'] = True
                elif parser_log['met_end_}'] and char == '"':
                    parser_log['code_end_index'] = back_index - 1
                    break
                else:
                    continue
            code_str = function_args[parser_log['code_begin_index']: parser_log['code_end_index'] + 1]
            if '\n' in code_str:
                return code_str.strip('\n')
            else:
                return json.loads(function_args)['code']

    except Exception as e:
        return None


if __name__ == "__main__":
    setup_logger(r"D:\GXQ\human_seg\neuron_seg_human\llm_neuron_seg\utils\log_utils\log.log")
    log_w_indent("This is a test question", 1, "Question")
    log_w_indent("This is a test response", 1, "Response")
