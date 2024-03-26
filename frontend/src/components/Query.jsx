import { CodeHighlight } from "@mantine/code-highlight"

function Query({ query }) {
    return (
      <CodeHighlight language="sql" code={query} withCopyButton={false} mb={20}/>
    )
  }
  
export default Query