import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    { text: " ¿Que cubre mi póliza?", value: " ¿Que cubre mi póliza?" },
    { text: " Tuve un siniestro, ¿que tengo que hacer?", value: " Tuve un siniestro, ¿que tengo que hacer?" },
    { text: " Me robaron, ¿que me cubre mi póliza?", value: "Me robaron, ¿que me cubre mi póliza?" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
